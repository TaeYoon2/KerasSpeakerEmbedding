import tensorflow as tf
class Batch_preprocess(tf.keras.layers.Layer):


    def __init__(self, sr, fft_hop, fft_window, nfft, nmel, mode, window, hop, **kwargs):
        self.sr = sr
        self.fft_hop = fft_hop / 1000
        self.fft_window = fft_window / 1000
        self.nfft = nfft
        self.nmel = nmel
        if mode == 'ms': # time in milisecond
            self.window = window / 1000
            self.hop = hop / 1000
            self.frame_per_win = int((self.window - self.fft_window)/self.fft_hop + 1)
            self.frame_per_hop = int((self.hop - self.fft_window)/self.fft_hop + 1)
        elif mode == 'frame':
            self.frame_per_win = self.window
            self.frame_per_hop = self.hop

        super(Batch_preprocess, self).__init__(**kwargs)

    def _mel(self, sig):
    	### batch mel extraction
        stfts = tf.signal.stft(sig, frame_length=int(self.fft_window * self.sr), frame_step=int(self.fft_hop * self.sr), fft_length=self.nfft)
        spectrograms = tf.abs(stfts)

        # Warp the linear scale spectrograms into the mel-scale.
        num_spectrogram_bins = stfts.shape[-1].value
        lower_edge_hertz, upper_edge_hertz = 80.0, 7600.0

        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(self.nmel, num_spectrogram_bins, self.sr, lower_edge_hertz, upper_edge_hertz)
        mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
        mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
        log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6) # [seq_length, num_mel]
        return log_mel_spectrograms

    def slice_bathces(self, list_mels):
        '''slice each mel of wavs and put them together'''
        # 자른 멜을 모을 리스트, 멜 사이즈 리스트 텐서
        batch_mels = []
        sizes_per_mel = None

        for list_mel in list_mels:
            lastidx = tf.shape(list_mel)[0] - self.frame_per_win + 1
            batch_mel = tf.map_fn(lambda i: tf.slice(list_mel, begin=tf.pad([i], paddings=[[0,1]]),size=[self.frame_per_win, self.nmel],name="sliding_window"), tf.range(lastidx, delta=self.frame_per_hop), dtype=tf.float32)
            size_per_mel = tf.shape(batch_mel)[:1]
            if sizes_per_mel is None:
                sizes_per_mel = size_per_mel
            else:
                sizes_per_mel = tf.concat([sizes_per_mel, size_per_mel],axis=0)
            batch_mels.append(batch_mel)
        # 자른 멜을 합친다.
        long_batch = tf.concat(batch_mels,axis=0)
        return (long_batch, sizes_per_mel)

    def call(self, inputs):
        sig = tf.squeeze(inputs[0],axis=0)
        sig_lengths = tf.squeeze(inputs[1],axis=0)
        mel_lengths = tf.map_fn(lambda x : (tf.math.floordiv(tf.math.subtract(x, int(self.fft_window*self.sr)), int(self.fft_hop*self.sr)))+1, sig_lengths)

        ragged_wavs = tf.RaggedTensor.from_row_lengths(values=sig, row_lengths=sig_lengths)
        sparse_wavs = ragged_wavs.to_tensor(default_value=0)
        sparse_mels = self._mel(sparse_wavs)
        ragged_mels = tf.RaggedTensor.from_tensor(sparse_mels, lengths=mel_lengths)
        size_splits = ragged_mels.row_lengths()
        list_mels = tf.split(ragged_mels.flat_values, mel_lengths, axis=0, num=size_splits[0], name="list_mels")

        # 배치 멜 + 길이 정보
        long_batch, sizes_per_mel = self.slice_bathces(list_mels)

        return (long_batch, sizes_per_mel)
