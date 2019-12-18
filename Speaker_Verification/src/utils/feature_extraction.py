import os
import librosa
import librosa.filters
import numpy as np
import scipy


class FeatureExtraction:
    """
    Note:
        compute STFT & log mel with librosa

    Attributes:
        __init__: constructs FeatureExtraction class
        process_signal: extract feature from a signal. run this
        _extract_wav_type_features: do all processes to a signal and return extracted feature
        _trim_silences: trim leading and trailing silence
        _stft: do STFT on a signal
        _mel_filter: apply mel filter to a stft output
        _transpose
        _log_clip_mel: clip mel
        _to_decibel
        _normalize_mel: normalize mel

    """

    def __init__(self, sr, nfft, fft_win_size, fft_hop_size, num_mel):
        """
        Note:

        Args:
            sr: integer, sampling rate
            nfft: integer, length of FFT window
            fft_win_size: integer, FFT window size (frames)
            fft_hop_size: integer, FFT hop size (frames)
            num_mel: integer, number of mel filters

        Returns:

        """

        self._sample_rate = sr
        self._nfft = nfft
        self._fft_window_size = fft_win_size
        self._fft_hop_size = fft_hop_size
        self._mel_bins = num_mel

        self._preemphasis_coef = 0.97
        self._fft_window_fn = "hann"
        self._fft_is_center = True
        self._mel_fmin = 0
        self._mel_fmax = 8000
        self._top_db = 60
        self._clip_mel = 1e-5
        # TODO : fix it clip linear
        self._clip_linear = 1e-5
        self._is_legacy = True
        self._min_db = -100
        self._max_db_normalize_range = 4
        self._min_db_normalize_range = -4

    def process_signal(self, sig):
        """
        Note:
            run _extract_wav_type_features according to self._is_linear

        Args:
            sig: np array, sound signal

        Returns:
            mel: np array, computed mel feature output

        """

        mel = np.empty(shape=[])

        mel, linear_spec = self._extract_wav_type_features(sig)
        return mel, linear_spec

    def _extract_wav_type_features(self, sig):
        """
        Note:
            extract log mel feature [, linear spectrogram feature] from a signal

        Args:
            sig: np array, sound signal

        Returns:
            _mel_spec: np array, computed mel feature output
            _spec: np array, linear (not mel filtered) signal

        """
        _wav = self._trim_silences(sig)
        # self._pre_emphasis()
        _spec = self._stft(_wav)
        _mel_spec = self._mel_filter(_spec)
        _mel_spec, _spec = self._transpose(_mel_spec), self._transpose(_spec)

        if self._is_legacy:
            _mel_spec, _spec = self._normalize(self._to_decibel(_mel_spec)), self._normalize(self._to_decibel(_spec))
        else:
            _mel_spec, _spec = self._log_clip(_mel_spec, self._clip_mel), self._log_clip(_mel_spec, self._clip_linear)

        return _mel_spec, _spec

    def _trim_silences(self, wav):
        """
        Note:
            trim leading and trailing silence under self._top_db

        Args:
            wav: np array, audio signal

        Returns:
            _wav: np array, trimmed signal

        """

        _wav, _ = librosa.effects.trim(wav, top_db=self._top_db, frame_length=2048, hop_length=512)
        return _wav

    def _stft(self, wav):
        """
        Note:
            STFT with librosa

        Args:
            wav: np array, sound signal

        Returns:
            _spec: np array, stft result

        """

        # short time fourier transformation
        _spec = librosa.stft(y=wav,
                             n_fft=self._nfft,
                             hop_length=self._fft_hop_size,
                             win_length=self._fft_window_size,
                             window=self._fft_window_fn,
                             center=self._fft_is_center,
                             pad_mode='reflect')
        _spec = np.abs(_spec)**2
        return _spec

    def _mel_filter(self, spec):
        """
        Note:
            apply mel filters

        Args:
            spec: np array, an output of STFT function

        Returns:
            _mel_spec: mel filtered STFT output

        """

        # Pass mel filter banks
        mel_filter_banks = librosa.filters.mel(sr=self._sample_rate, n_fft=self._nfft,
                                               n_mels=self._mel_bins, fmin=self._mel_fmin, fmax=self._mel_fmax)
        _mel_spec = np.dot(mel_filter_banks, spec)
        return _mel_spec

    def _transpose(self, spec):
        """
        Note:
            transpose feature

        Args:
            spec: np array, feature array

        Returns:
            _spec: transposed feature

        """

        _spec = np.transpose(spec, (1, 0))

        return _spec

    def _log_clip(self, spec, clip):
        """
        Note:
            clip by self._clip_mel and transform mel into log mel

        Args:
            spec: np array, computed mel filtered feature

        Returns:
            _log_clipped_spec: np array, log clipped feature

        """

        _clipped_spec = np.clip(spec, clip, None)
        _log_clipped_spec = np.log(_clipped_spec)

        return _log_clipped_spec

    def _to_decibel(self, spec):
        """
        Note:
            convert mel feature to decibel scale

        Args:
            spec: np array, spectrogram(output of stft~)

        Returns:
            _db_spec: np array, spectrogram in decibel

        """

        min_level = np.exp(-100/20*np.log(10))
        _db_spec = 20*np.log10(np.maximum(min_level, spec))-20

        return _db_spec

    def _to_amplitude(self, decibel):
        """
        Note:
            convert decibel scale to origin

        Args:
            decibel: np array, decibel

        Returns:
            _db_mel_spec: np array, mel spectrogram in decibel

        """
        return np.power(10, decibel * (1/20))

    def _normalize(self, db_spec):
        """
        Note:
            normalize spectrogram feature

        Args:
            db_spec: np array, spectrogram in decibel

        Returns:
            _normalized_db_spec: np array, normalized spectrogram in decibel

        """

        _normalized_db_spec = np.clip(2*self._max_db_normalize_range*((db_spec-self._min_db)/-self._min_db)-self._max_db_normalize_range, self._min_db_normalize_range, self._max_db_normalize_range)

        return _normalized_db_spec
