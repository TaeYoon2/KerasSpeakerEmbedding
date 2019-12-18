import os
import glob
import librosa
import numpy as np
from multiprocessing import Pool
import shutil
from tqdm import tqdm
import re
# utils files
from feature_extraction import FeatureExtraction
from vad import *

# TODO 여러 DB 처리할 때, 순서대로 중첩되어 저장됨
class Wav2Mel(object):
    """
    Note:
        Convert audio signal/file to mel feature
        VAD & feature extraction process
    
    Attributes:
        __init__: constructs Wav2Mel class, initializes FeatureExtraction and VAD classes
        process_db: processes all wav files under input directory path and save .npy to designated output path
        wav_to_mel: processes input signal and returns mel array
        _vad: VAD process
        _feature_extraction: log mel feature extraction process
        _one_wav_for_process_db: called in process_db, processes one wav file path and save a .npy
        _get_audio_list: get list of .wav or .mp4 excluding that of speakers(folders) which contain less than 10(or defined number of) utts
        _remove_spkr_under_numutt: in process_db, after saving .npy remove directories that contain less than 10(or defined number of) utts
        _save: check directory and save .npy
        _make_dir_tree: when process_db with multiprocessing option, make all save dir tree in advance
        _dir_sanity_check: directory path filtering
    """

    def __init__(self, min_num_utt_per_spkr, min_utt_duration, sample_rate, num_fft, fft_window_size, fft_hop_size, num_mel, vad_mode, preprocess_multiprocessing=False):
        """
        Note:
        
        Args:
            min_num_utt_per_spkr: integer, minimum number of utterances per speaker
            min_utt_duration: float, minimum duration of an audio file (in frames)
            sample_rate: integer, sampling rate
            num_fft: integer, fft window size
            fft_window_size: integer, The window will be of length win_length and then padded with zeros to match n_fft (msec)
            fft_hop_size: integer, number audio of frames between STFT columns (msec)
            num_mel: integer, feature dimension
            vad_mode: integer ranging [1,3]. degree of strictness in VAD
            multiprocessing: Boolean, True when processing large DB and saving it to .npy

        Returns:
            
        """

        ### General settings
        self._min_num_utt_per_spkr = min_num_utt_per_spkr
        self._min_utt_duration = (((min_utt_duration-1)*fft_hop_size)+fft_window_size)/1000
        self._min_utt_duration_fr = min_utt_duration

        ### VAD
        self.vad = VAD(vad_mode, "unused")

        ### FFT
        fft_window_size_frames = int((fft_window_size / 1000) * sample_rate)
        fft_hop_size_frames = int((fft_hop_size / 1000) * sample_rate)
        self.feature_extractor = FeatureExtraction(sample_rate, num_fft, fft_window_size_frames, fft_hop_size_frames, num_mel)
        self._multiprocessing = preprocess_multiprocessing


    def process_db(self, db_dir, mel_dir):
        """
        Note:
            does VAD & feature extraction of wav files from a directory and save processed mel features as .npy
            uses Pool if multiprocessing=True in __init__ config

        Args:
            db_dir : string that indicates a directory of the original DB
                     A wav file should be directely under its speaker folder
            mel_dir : string that indicates a directory where the processed mels are saved as npy
                      its sub-directories are the same as db_dir

        Returns:

        """

        db_dir = self._dir_sanity_check(db_dir, isexist=True)
        mel_dir = self._dir_sanity_check(mel_dir)
        wav_list, spkr_list = self._get_audio_list(db_dir)
        if self._multiprocessing:
            self._make_dir_tree(spkr_list, db_dir, mel_dir)
            db_dir_param = [db_dir]*len(wav_list)
            mel_dir_param = [mel_dir]*len(wav_list)
            params = zip(wav_list, db_dir_param, mel_dir_param)
            with Pool(processes=4) as pool:
                pool.map(self._one_wav_for_process_db, params)
        else:
            for wav_file in tqdm(wav_list):
                params = (wav_file, db_dir, mel_dir)
                self._one_wav_for_process_db(params)
        self._remove_spkr_under_numutt(mel_dir)

    def wav_to_mel(self, wav_file):
        """
        Note:
        does VAD & feature extraction of a wav file and return processed mels as an np array
            
        Args:
        wav_file: a string that indicates path of a .wav file

        Returns:
        mel: a np array of processed mel feature
        
        """

        sig, sr = librosa.load(wav_file, sr=None)
        wav_dur = len(sig)/sr
        if wav_dur < self._min_utt_duration:
            print(f"skipping RAW {wav_file} {wav_dur} < {self._min_utt_duration}")
            return []
        else:
            processed_sig = self._vad(wav_file)
            if len(processed_sig)/sr < self._min_utt_duration:
                print(f"skipping VAD output {wav_file} {len(processed_sig)/sr}< {self._min_utt_duration}")
                return []
            else:
                mel, _ = self._feature_extraction(processed_sig)
                return mel
                                                                                                 
    def _vad(self, wav_file):
        """
        Note:
            does VAD from a file and return the result as an array

        Args:
            wav_file: a string that indicates path of a .wav file

        Returns:
            total_wav: an array of the VAD processed signal

        """

        total_wav, sample_rate = self.vad.run_vad(wav_file)
        return total_wav

    def _feature_extraction(self, sig):
        """
        Note:
            extracts log mel feature from a signal

        Args:
            sig: an array of input audio signal

        Returns:
            out_mel: an array of mel features

        """
        out_mel = self.feature_extractor.process_signal(sig)
        return out_mel


    def _one_wav_for_process_db(self, params):
        """
        Note:
            does VAD & feature extraction of a wav file and save the processed mel array to a new directory as .npy file
            this is designed for multiprocessing Pool

        Args:
            params: tuple of (wav_file, db_dir, mel_dir), strings of paths

        Returns:

        """

        wav_file = params[0]
        db_dir = params[1]
        mel_dir = params[2]
        fn = re.sub(db_dir,mel_dir,wav_file)
        fn = re.sub(r'\.[^/]+','.npy',fn)
        new_path = os.path.join(mel_dir, fn)
        print(new_path)
        mel = self.wav_to_mel(wav_file)
        if len(mel)>0:
            self._save(mel, new_path)

    def _get_audio_list(self, db_dir):
        """
        Note:
            glob .wav or .m4a(add more extensions if needed) files from input directory
            excludes utterances of which the speaker contains less than min_num_utt_per_spkr

        Args:
            db_dir: a string that indicates top directory containing audio files 

        Returns:
            out_wav_list: a list of strings which are paths of each audio file
            uniq_spkr: a list of strings which are directories representing each speaker
                       assuming direct upper directory of audio files are speaker directories

        """

        wav_list = []
        ext = ['wav', 'm4a']
        for e in ext:
            print(f"getting all {e} under {db_dir}")
            wav_list.extend(glob.glob(os.path.join(db_dir, '**/*.'+e), recursive=True))
        spkr_path_list = ['/'.join(i.split('/')[:-1]) for i in wav_list]
        uniq_spkr = list(set(spkr_path_list))
        out_wav_list = []
        for spkr_path in uniq_spkr:
            indices = [i for i, x in enumerate(spkr_path_list) if x == spkr_path]
            if len(indices) >= self._min_num_utt_per_spkr:
                out_wav_list.extend([wav_list[i] for i in indices])
        return out_wav_list, uniq_spkr

    def _remove_spkr_under_numutt(self, mel_dir):
        """
        Note:
            remove speaker directories that contain less than min_num_utt_per_spkr

        Args:
            mel_dir: a string of a directory path

        Returns:

        """

        npy_list = glob.glob(os.path.join(mel_dir, '**/*.npy'), recursive=True)
        spkr_path_list = ['/'.join(i.split('/')[:-1]) for i in npy_list]
        uniq_spkr = list(set(spkr_path_list))
        for spkr_path in uniq_spkr:
            indices = [i for i, x in enumerate(spkr_path_list) if x == spkr_path]
            if len(indices) < self._min_num_utt_per_spkr:
                shutil.rmtree(spkr_path)
                print(f"{spkr_path} is erased since it contains less than {self._min_num_utt_per_spkr} npy files")
        return

    def _save(self, arr, path):
        """
        Note:
            save an array with np.save

        Args:
            arr: numpy array that is to be saved
            path: a string that indicates a path where .npy will be saved

        Returns:

        """

        save_dir = os.path.dirname(path)
        if not os.path.exists(save_dir):
            if not self._multiprocessing:
                os.makedirs(save_dir)
            else:
                print(f"directory {save_dir} is not prepared, check self._make_dir_tree")
        #print(arr)
        np.save(path, arr)
        print(arr.shape)
        return

    def _make_dir_tree(self, spkr_dir_list, db_root, mel_root):
        """
        Note:
            make all directory tree for saving .npy files in advance
            this is for multiprocessing=True, because Pool tends to yield an error with os.path.exists & os.makedirs combination

        Args:
            spkr_dir_list: list of strings that indicate speaker folders from source database directory
            db_root: a string that indicates the top directory path of the source database
            mel_root: a string that indicates the top directory path of target saving location 

        Returns:

        """

        mel_dir_list = [re.sub(db_root, mel_root, x) for x in spkr_dir_list]
        for mel_dir in mel_dir_list:
            os.makedirs(mel_dir)
        return

    def _dir_sanity_check(self, path, isexist=False):
        
        """
        Note:
            deletes ending slash character that triggers error with os.path.join
            checks if directory exists if isexist=True

        Args:
            path: a string of a path
            isexist: if isexist=True, checks if the input path exists
                     default value is False

        Returns:
            path: a string of cleaned path

        """

        if isexist:
            if not os.path.exists(path):
                print(f"{path} does not exist; aborted")
                exit()
        if path[-1] == '/':
            path = path[:-1]
        return path


if __name__=="__main__":

    min_utt_ct = 10
    min_utt_fr = 160
    sample_rate = 8000
    nfft = 2048
    fft_win_size = 25
    fft_hop_size = 10
    nmel = 40
    vad_mode = 1
    
    # db_root = "/mnt/nas/03_ETC/voxceleb/vox1/test"
    db_root = "/mnt/data1/sunkist/data/KOR-CLN-V1-8k"
    # save_dir = "/home/sh77/data/speaker_id/npy_tmp"
    save_dir = "/mnt/data1/sunkist/data/KOR-CLN-V1-8k-mel"
    processor = Wav2Mel(min_utt_ct, min_utt_fr, sample_rate, nfft, fft_win_size, fft_hop_size, nmel, vad_mode)

    for i in ['01_DICT/1','02_HTST/1','05_IHMC/1','06_KSCO/1','07_ERD2/1','08_ERD3/1','09_MOBI/1','11_RSPC/1','12_ESP1/1','14_ERD1/1']:        
        db = os.path.join(db_root,i)
        print(f"PREPROCESSING DB {i}")
        save_dir=os.path.join(save_dir,i)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        processor.process_db(db, save_dir)
