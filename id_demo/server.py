import tornado.ioloop
import tornado.web
import tornado.httpserver
import json
import numpy as np
import sys
import os
import pickle
import requests
import re
from operator import itemgetter
from tornado.options import define, options
import librosa
import soundfile as sf
import glob
from time import gmtime, strftime

sf.default_subtype('WAV')


sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from Speaker_Verification.src.utils.infer_process import InferPrep


# config update
CONFIG_FILE = os.path.abspath(os.path.join(os.pardir,'Speaker_Verification','config', 'config.json'))
with open(CONFIG_FILE, 'rb') as fid:
    config = json.load(fid)
config_update = {
    "preemphasis_coef": 0.97,
    "fft_window_fn": "hann",
    "fft_is_center": True,
    "is_legacy": False,
    "is_linear": False
    }
config.update(config_update)


define("certfile", type=str)
define("keyfile", type=str)
define("api_port", type=str)
define("serving_port",type=str)
define("serving_host",type=str)
define("dvec_dir", type=str)
define("wav_root", type=str)

tornado.options.parse_config_file("config.cfg")
with open(options.dvec_dir, 'rb') as f:
    emb_dict = pickle.load(f)

def cos_sim(mat, emb):
    sim_mat = np.dot(mat, emb)/(np.linalg.norm(mat, axis=1)*np.linalg.norm(emb))
    return sim_mat

class MainHandler(tornado.web.RequestHandler):

    def initialize(self):
        self.mel_savedir="./tmp"
        if not os.path.exists(self.mel_savedir):
            os.makedirs(self.mel_savedir)
        self.infer_processor = InferPrep(config,self.mel_savedir)

    # @tornado.web.asynchronous
    def get(self):
        self.render("static/templates/index.html")

    def post(self):
        pcmdata=self.request.body
        try:
            data = np.fromstring(pcmdata, dtype=np.int16)
            data = np.float32(data/32768)

            #temporary timestamp solution to prevent wav-saving overwriting between users
            timestamp=strftime("%Y%m%d%H%M%S", gmtime())
            # save received audio as 'mel_savedir/in.wav'
            sf.write(os.path.join(self.mel_savedir,f'{timestamp}.wav'), data, 16000)

            # convert wav to mel (time wise batch)
            batch_mel = self.infer_processor.run_preprocess_serving(os.path.join(self.mel_savedir,f'{timestamp}.wav'))
  
            # send api request
            r = requests.post(f"http://{options.serving_host}:{options.serving_port}/v1/models/ge2e:predict",
                              json={"inputs" :  batch_mel.tolist() })
            print(r.status_code)
            
            embedding = r.json()["outputs"]

            sim_mat = cos_sim(emb_dict['dvectors'], embedding[0])
            max_idx = sim_mat.argsort()[::-1][:5]
            spk_list = itemgetter(*max_idx)(emb_dict['labels'])
            spk_list_flac = [ re.sub('.wav','.flac',i) for i in spk_list ]

            json_make = {
                "payload" : {
                    "result" : spk_list_flac,
                },
                "event" : "result"
            }
            json_final = json.dumps(json_make)
            self.write(json_final)
        except Exception as e:
            print("except", e)
            json_make = {
                "payload" : {},
                "event" : "no_result"
            }
            json_final = json.dumps(json_make)
            self.write(json_final)

class WavHandler(tornado.web.RequestHandler):
    # @tornado.web.asynchronous
    def post(self):
        json_data = json.loads(self.request.body)
        #wpath = os.path.join(options.wav_root, json_data['path'])
        wpath = glob.glob(options.wav_root+'/train-clean-*/*/*/'+json_data['path'])[0]
        print(f"wpath {wpath}")
        print('received wav file request: '+ wpath)
        chunk_size = 1024 * 10
        with open(wpath, 'rb') as f:
            while 1:
                data = f.read(chunk_size) # or some other nice-sized chunk
                if not data: break
                self.write(data)
        self.finish()
        return

def make_app():
    return tornado.web.Application([
        (r"/", MainHandler),
        (r"/wav", WavHandler),
        (r'/static/(.*)', tornado.web.StaticFileHandler,{"path": "./static"})
    ])

if __name__ == "__main__":
    app = make_app()
    server = tornado.httpserver.HTTPServer(app, ssl_options={
        "certfile": options.certfile,
        "keyfile": options.keyfile})
    server.listen(options.api_port)

    tornado.ioloop.IOLoop.current().start()
