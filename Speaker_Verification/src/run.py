import os
import json
import argparse
import configparser
import warnings
import datetime
from ge2e import *
warnings.simplefilter(action='ignore', category=FutureWarning)

# arguments
# ckpt example : '/path/to/your/ckpt/cp-{:06d}.ckpt'
parser = argparse.ArgumentParser()
parser.add_argument("--mode")
parser.add_argument("--ckpt",default='/mnt/data1/sunkist/data/sv_ckpt/kor_8k/cp-{:06d}.ckpt')
parser.add_argument("--infer_json",default='')
parser.add_argument("--wavdir",default='')
parser.add_argument("--emb_savedir",default='/mnt/data1/sunkist/data/kor_cln_test_emb')
args = parser.parse_args()

def get_data_list(data_type, root_dir=''):
    datalist = []
    if data_type == "wav":
        if root_dir:
            datalist = glob.glob(root_dir+'/*wav')
    elif data_type == "mel":
        if root_dir:
            datalist = glob.glob(root_dir+'/*npy')  #?
    elif data_type == "testset":
        with open(os.path.join(os.path.dirname(__file__),'path_index_test.json'), 'r') as f:
            utts = json.load(f)['utts']
        for spk in utts.keys():
            datalist.extend(utts[spk])
    return datalist

if __name__ == "__main__":
    mode = args.mode
    ckpt_to_load = args.ckpt

    # config
    CONFIG_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'config', 'config.json'))
    with open(CONFIG_FILE, 'rb') as fid:
        config = json.load(fid)
    # config 읽어와서 후처리
    config["write_wavname"] = bool(config["write_wavname"])
    config["random_segment"] = bool(config["random_segment"])
    config["multiprocessing"] = bool(config["multiprocessing"])
    config["preprocess_multiprocessing"] = bool(config["preprocess_multiprocessing"])

    if mode == "training":
        sv = Speaker_verification(config,"train")
    elif mode == "inference":
        sv = Speaker_verification(config,"infer")
    elif mode == "evaluation":
        sv = Speaker_verification(config,"infer")
    elif mode == "exporting":
        sv = Speaker_verification(config,"export")


    if mode == "train":
        sv._run()
    elif mode == "infer":
        emb_savedir= args.emb_savedir

        if not os.path.exists(emb_savedir):
            os.makedirs(emb_savedir)

        wavepath_list = get_data_list("wav", args.wavdir)
        # process wav & infer
        sv._process_and_infer(config,emb_savedir,ckpt_to_load,wavpath_list)

    elif mode == "evaluation":
        evaluation_info = []
        emb_savedir= args.emb_savedir

        if not os.path.exists(emb_savedir):
            os.makedirs(emb_savedir)

        mellist = get_data_list("testset")

        for i in range(79,6060):
            each_checkpoint = ckpt_to_load.format(i)
            # process mel & infer
            timenow, num_epoch, eer = sv._infer_mels(config,emb_savedir,each_checkpoint,mellist)
            evaluation_info.append([timenow, num_epoch, eer])
        
        for item in evaluation_info:
            print(f"[{item[0]}] EPOCH : {item[1]},  EER : {eer}")

    elif mode == "exporting":
        sv._export(ckpt_to_load)
