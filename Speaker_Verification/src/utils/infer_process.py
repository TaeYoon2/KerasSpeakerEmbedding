import numpy as np
import os
import re
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from wav_to_mel import Wav2Mel
import pdb



class InferPrep(object):
    def __init__(self,config):
        self.window = config["ge2e_window_size"]
        self.overlap = config["ge2e_hop_size"]
        self.fft_window = config["fft_window_size"]
        self.fft_hop = config["fft_hop_size"]

        self.processor = Wav2Mel( config["min_num_utt_per_spkr"],
                                  config["min_utt_duration"],
                                  config["sample_rate"],
                                  config["num_fft"],
                                  config["fft_window_size"],
                                  config["fft_hop_size"],
                                  config["num_mel"],
                                  config["vad_mode"],
                                  config["preprocess_multiprocessing"])


        return

    def window_and_stack_batch(self,mel_input,window_size,fft_window_size,fft_hop,overlap,window_unit='frame'):
        mode = window_unit
        if mode == 'ms': # time in milisecond
            frame_per_win = int((window_size - fft_window_size)/fft_hop + 1)
            frame_per_hop = int((overlap - fft_window_size)/fft_hop + 1)
        elif mode == 'frame':
            frame_per_win = window_size
            frame_per_hop = overlap
        batch = []
        i=0
        while True:
            if i+frame_per_win>mel_input.shape[0]:
                break
            batch.append(mel_input[i:i+frame_per_win][:])
            i+=frame_per_hop
        if len(batch) != 0:
            stacked_batch = np.stack(batch,axis=0)
        else:
            stacked_batch=[]
        
        return stacked_batch

    def run_preprocess_infer(self,wavpath):
        # if wav_or_melnpy_path.split('.')[-1]=='wav':
        #     basename=wav_or_melnpy_path.split('/')[-1]
        #     dbname=wav_or_melnpy_path.split('/')[-4]
        #     npy_name = 'melbatch_{}_{}'.format(dbname,re.sub(".wav",".npy",basename))
        mel = self.processor.wav_to_mel(wavpath)
                                     
        # elif wav_or_melnpy_path.split('.')[-1]=='npy':
        #     npy_name=wav_or_melnpy_path
        #     mel = np.load(wav_or_melnpy_path)
            
        if isinstance(mel,np.ndarray):
            batch_mel = self.window_and_stack_batch(mel,self.window,self.fft_window,self.fft_hop,self.overlap)
            if batch_mel != []:
                return batch_mel
            else:
                return []
        else:
            return []
                        
    def run_preprocess_serving(self,wavpath):
        npy_name = 'melbatch_' + re.sub(".wav",".npy",os.path.basename(wavpath))
        mel = self.processor.wav_to_mel(wavpath)
        if mel != []:
            batch_mel = self.window_and_stack_batch(mel,self.window,self.fft_window,self.fft_hop,self.overlap)
            #np.save(os.path.join(self.infer_preprocess_savedir,npy_name),batch_mel)

        return batch_mel



