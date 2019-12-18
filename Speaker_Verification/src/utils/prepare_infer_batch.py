import os
import glob
import pool
from utils.infer_process import InferPrep



def prep_infer_mel_batches_from_wav(config, wav_path, emb_save_path):
    # give args.wavdir so that [ args.wavdir+'/*wav' ] can capture all the wavfiles you intend to infer
    mel_save_path = emb_save_path + '_preprocessed'
    if not os.path.exists(mel_save_path):
        os.makedirs(mel_save_path)

    wavlist = glob.glob(wav_path+'/*wav')
    emb_done_list = [os.path.basename(i) for i in glob.glob(emb_save_path+'/*npy')]
    # filter out wavfiles that are already preprocessed
    wavlist_to_preprocess = [ i for i in wavlist if re.sub('wav', 'npy', os.path.basename(i)) not in emb_done_list]
    print(f"todo wavlist: {len(wavlist_to_preprocess)} of {len(wavlist)}")
    
    # config update for Inference preprocessing
    config_update = {
        "preemphasis_coef": 0.97,
        "fft_window_fn": "hann",
        "fft_is_center": True,
        "is_legacy": False,
        "is_linear": False
    }
    config.update(config_update)
    pool=Pool(3)
    inferprep=InferPrep(config,mel_save_path)
    pool.map(inferprep.run_preprocess_infer, wavlist_to_preprocess)
    print(f"finished preprocessing to dir {mel_save_path}")
    return mel_save_path


