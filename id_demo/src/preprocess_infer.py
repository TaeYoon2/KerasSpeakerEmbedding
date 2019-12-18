import numpy as np
import tensorflow as tf
import os
import librosa


class PrepareInfer():
    def __init__(self, fft_window=25, fft_hop=10, nfft=2048, nmel=40, window=800, overlap=400, window_unit='ms'):
        #### parameters
        # parameters for stft
        self.fft_window = fft_window / 1000 #0.025 (convert ms to sec)
        self.fft_hop = fft_hop / 1000 #0.01
        self.nfft = nfft #2048
        self.nmel = nmel #40
        
        # parameters for windowing(unit for extracting one embedding vector)
        self.window_unit = window_unit
        if self.window_unit == 'ms':  # unit for multitaco: ms(800) / for GE2E alone: frame(160)
            self.window = window / 1000
            self.overlap = overlap / 1000
        else:
            self.window = window
            self.overlap = overlap
        return

    def feature_extraction_tf(self, pcm, sr):
        self.pcm, self.sr = pcm, sr
        hop, window, nfft, num_mel_bins = self.fft_hop, self.fft_window, self.nfft, self.nmel

        sig = tf.placeholder(dtype=tf.float32,shape=pcm.shape)        
        # sig = tf.cast(sig, tf.float32)
        stfts = tf.signal.stft(sig, frame_length=int(window * sr), frame_step=int(hop * sr), fft_length=nfft)
        spectrograms = tf.abs(stfts)

        # Warp the linear scale spectrograms into the mel-scale.
        num_spectrogram_bins = stfts.shape[-1].value
        lower_edge_hertz, upper_edge_hertz = 80.0, 7600.0

        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(num_mel_bins, num_spectrogram_bins, sr, lower_edge_hertz, upper_edge_hertz)
        mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)

        mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
        print('mel_spectrograms after: ', tf.shape(mel_spectrograms))
        log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)

        mode = self.window_unit
        if mode == 'ms': # time in milisecond
            frame_per_win = int((self.window - self.fft_window)/self.fft_hop + 1)
            frame_per_hop = int((self.overlap - self.fft_window)/self.fft_hop + 1)
        elif mode == 'frame':
            frame_per_win = self.window
            frame_per_hop = self.overlap

        # def extract_patches(x):
        #     return tf.extract_image_patches(
        #         x,
        #         (1, frame_per_win, 40, 1),
        #         (1, frame_per_hop, 1, 1),
        #         (1, 1, 1, 1),
        #         padding="VALID"
        #     )
        # log_mel_spectrograms=tf.expand_dims(log_mel_spectrograms,0)
        # log_mel_spectrograms=tf.expand_dims(log_mel_spectrograms,-1)
        # batch = extract_patches(log_mel_spectrograms)

        batch = tf.map_fn(lambda i: log_mel_spectrograms[i:i+frame_per_win], tf.range(log_mel_spectrograms.shape[0]-frame_per_win+1,delta=frame_per_hop), dtype=tf.float32)

        with tf.compat.v1.Session() as sess:
            logmel_batch = sess.run(batch, feed_dict={sig:pcm})
            
        return logmel_batch

    
    def _feature_extraction_from_array(self, sig, sr):
        self.sig,self.sr = sig, sr
        hop, window, nfft, num_mel_bins = self.fft_hop, self.fft_window, self.nfft, self.nmel
        print("start feature extraction")
        stfts = librosa.stft(sig, n_fft=nfft, hop_length=int(hop*sr), win_length=int(window*sr))
        D = np.abs(stfts)**2
        mel_spectrograms = librosa.feature.melspectrogram(S=D, n_mels=num_mel_bins)
        mel_spectrograms = mel_spectrograms.T
        print('mel_spectrograms after: ', mel_spectrograms.shape)

        # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
        log_mel_spectrograms = np.log(mel_spectrograms + 1e-6)
        return log_mel_spectrograms
    
    def _windowing(self, frames, mode='ms'):
        mode = self.window_unit
        if mode == 'ms': # time in milisecond
            frame_per_win = int((self.window - self.fft_window)/self.fft_hop + 1)
            frame_per_hop = int((self.overlap - self.fft_window)/self.fft_hop + 1)
        elif mode == 'frame':
            frame_per_win = self.window
            frame_per_hop = self.overlap
        batch = []
        i=0
        while True:
            if i+frame_per_win>frames.shape[0]:
                break
            batch.append(frames[i:i+frame_per_win][:])
            i+=frame_per_hop
        stacked_batch = np.stack(batch,axis=0)
        print('final batch shape: ', stacked_batch.shape)
        return stacked_batch

    def prepare_infer(self, sigarr, sr):
        features = self._feature_extraction_from_array(sigarr, sr)
        windowed = self._windowing(features)
        return windowed


if __name__=="__main__":
    wavfn="/mnt/nas/01_ASR/01_korean/01_speech_text/KOR-CLN-V1/01_DICT/1/f2_jeo_315/f2_jeo_315_0096.wav"
    sig, sr = librosa.load(wavfn, sr=None)
    pp = PrepareInfer()
    mel_batch = pp.feature_extraction_tf(sig, sr)
    print(mel_batch.shape)

    librosa_batch = pp.prepare_infer(sig,sr)
    print(librosa_batch.shape)
