import tensorflow as tf
class Preprocess(tf.keras.layers.Layer):
    """
    Note:
        Prepare a batch of mels from a wave signal by tf.signal API when inferencing

    Attributes:
        __init__: constructs Ge2e_loss class
        call: compute the ge2e loss
    """


    def __init__(self, sr, fft_hop, fft_window, nfft, nmel, mode, window, hop, **kwargs):
        """
        Note:
            set up the preprocess configurations;
        Args:
            sr: the sample rate of a wav signal to be fed
            fft_hop: the hopping size of fft
            fft_window: the window size of fft to be looked at
            nfft: the number of samples to be looked at while fft;
                   2 times the frequencies to be looked at by fft
            nmel: the number of dimesion in mel features
            mode: how the above arguments to be fed; 'ms' or 'frame'
            window: the window size of ge2e-sliding-window at mel frames
            hop: the hop size of ge2e-sliding-window at mel frames

        Returns:

        """

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
            self.window = window
            self.hop = hop
            self.frame_per_win = self.window
            self.frame_per_hop = self.hop

        super(Preprocess, self).__init__(**kwargs)

    def call(self, sig):
        """
        Note:
            pass fft & melfilter bank & ge2e-window-slicing on wave singal
        Args:
            sig: a random length 1D[1,1] wave signal

        Returns:
            batch: a batch for a inference
        """

        sig = tf.keras.backend.squeeze(sig, axis=0)
        stfts = tf.signal.stft(sig, frame_length=int(self.fft_window * self.sr), frame_step=int(self.fft_hop * self.sr), fft_length=self.nfft)
        spectrograms = tf.abs(stfts)

        # Warp the linear scale spectrograms into the mel-scale.
        num_spectrogram_bins = stfts.shape[-1].value
        lower_edge_hertz, upper_edge_hertz = 80.0, 7600.0

        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(self.nmel, num_spectrogram_bins, self.sr, lower_edge_hertz, upper_edge_hertz)
        mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
        mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
        # [seq_length, num_mel]
        log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
        lastidx = tf.shape(log_mel_spectrograms)[0] - self.frame_per_win + 1
        batch = tf.map_fn(lambda i: tf.slice(log_mel_spectrograms, begin=tf.pad([i], paddings=[[0,1]]),size=[self.frame_per_win, self.nmel],name="sliding_window"), tf.range(lastidx, delta=self.frame_per_hop), dtype=tf.float32)

        return batch
