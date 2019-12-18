import webrtcvad
import collections
import contextlib
import wave
import numpy as np
import os
import glob
from pydub import AudioSegment


class VAD(object):
    """
    Note:
        VAD using webrtcvad https://github.com/wiseman/py-webrtcvad

    Attributes:
        __init__: constructs VAD class
    """

    def __init__(self, vad_mode, save_dir):
        """
        Note:

        Args:
            vad_mode: integer between 0 and 3, aggressiveness of filtering out non-speech
            save_dir: string, path to save vad result signal

        Returns:

        """

        self.vad_mode = vad_mode
        self.outdir = save_dir
        self.vad = webrtcvad.Vad(self.vad_mode)

    def run_vad(self, path):
        """
        Note:
            run VAD from a single file

        Args:
            path: string, a path of a single audio file

        Returns:
            total_wav: string, binary encoded result of VAD
            sample_rate: integer, sampling rate in Hz

        """

        if path.split('.')[-1]=='wav':
            audio, sample_rate = self.read_wave(path)
        elif path.split('.')[-1]=='m4a':
            audio, sample_rate = self.read_m4a(path)
        elif path.split('.')[-1]=='flac':
            audio, sample_rate = self.read_flac(path)

        frames = self.frame_generator(30, audio, sample_rate)
        segments = self.vad_collector(sample_rate, 30, 300, self.vad, frames)
        total_wav = b""
        for i, segment in enumerate(segments):
            total_wav += segment

        # 16bit PCM 기준 dtype=np.int16
        wav_arr = np.frombuffer(total_wav, dtype=np.int16)
        float_wav = np.float32(wav_arr) / 32768  # convert int16 pcm to float

        return  float_wav, sample_rate

    def run_vad_list(self, path_list):
        """
        Note:
            run_vad over a list of multiple paths and save the output as a wav file

        Args:
            path_list: list of strings, each of them is a path of an audio file

        Returns:
        """
        if type(path_list) == str:
            path_list = [path_list]
        assert type(path_list) == list

        for path in path_list:
            total_wav = self.run_vad_single(path)

            subpath = '/'.join(path.split('/')[-3:])
            subdir = '/'.join(path.split('/')[-3:-1])
            outpath = os.path.join(self.outdir,subpath)
            outdir = os.path.join(self.outdir,subdir)

            if not os.path.exists(outdir):
                os.makedirs(outdir)
            self.write_wave(outpath, total_wav, sample_rate)  #path, audio, sample_rate

        return

    def convert_to_float(self, binary_wav):
        """
        Note:
            convert binary 16bit encoded string to float array

        Args:
            binary_wav: string, binary encoded audio

        Returns:
            float_wav: np array, float audio signal

        """
        # 16bit PCM 기준 dtype=np.int16
        wav_arr = np.frombuffer(binary_wav, dtype=np.int16)
        float_wav = np.float32(wav_arr) / 32768  # convert int16 pcm to float
        return float_wav

    def read_wave(self, path):
        """
        Note:
            this is from webrtcvad example.py

        Args:
            path: string, a path of a wav file

        Returns:
            pcm_data: string, binary wav
            sample_rate: integer, sampling rate

        """

        with contextlib.closing(wave.open(path, 'rb')) as wf:
            num_channels = wf.getnchannels()
            assert num_channels == 1
            sample_width = wf.getsampwidth()
            assert sample_width == 2
            sample_rate = wf.getframerate()
            assert sample_rate in (8000, 16000, 32000)
            pcm_data = wf.readframes(wf.getnframes())
            return pcm_data, sample_rate

    def read_m4a(self,path):
        """
        Note:

        Args:
            path: string, a path of a mp4 file

        Returns:
            pcm_data: string, binary wav
            sample_rate: integer, sampling rate

        """

        mf = AudioSegment.from_file(path, "m4a")
        sample_rate = mf.frame_rate
        pcm_data = mf.raw_data
        return pcm_data, sample_rate

    def read_flac(self,path):
        """
        Note:

        Args:
            path: string, a path of a flac file

        Returns:
            pcm_data: string, binary wav
            sample_rate: integer, sampling rate

        """

        mf = AudioSegment.from_file(path, "flac")
        sample_rate = mf.frame_rate
        pcm_data = mf.raw_data
        return pcm_data, sample_rate


    def frame_generator(self, frame_duration_ms, audio, sample_rate):
        """
        Note:
            from webrtcvad example.py
            original note:
                Generates audio frames from PCM audio data.
                Takes the desired frame duration in milliseconds, the PCM data, and
                the sample rate.
                Yields Frames of the requested duration.

        Args:
            frame_duration_ms: The frame duration in milliseconds
            audio: loaded binary sound signal
            sample_rate: The audio sample rate, in Hz

        Returns:
            Frames of the requested duration

        """

        n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
        offset = 0
        timestamp = 0.0
        duration = (float(n) / sample_rate) / 2.0
        while offset + n < len(audio):
            yield Frame(audio[offset:offset + n], timestamp, duration)
            timestamp += duration
            offset += n

    def vad_collector(self, sample_rate, frame_duration_ms,
                      padding_duration_ms, vad, frames):
        """
        Note:
            from webrtcvad example.py
            original note:
                Filters out non-voiced audio frames.
                Given a webrtcvad.Vad and a source of audio frames, yields only
                the voiced audio.
                Uses a padded, sliding window algorithm over the audio frames.
                When more than 90% of the frames in the window are voiced (as
                reported by the VAD), the collector triggers and begins yielding
                audio frames. Then the collector waits until 90% of the frames in
                the window are unvoiced to detrigger.
                The window is padded at the front and back to provide a small
                amount of silence or the beginnings/endings of speech around the
                voiced frames.

        Args:
            sample_rate: The audio sample rate, in Hz
            frame_duration_ms: The frame duration in milliseconds
            padding_duration_ms: The amount to pad the window, in milliseconds
            vad: An instance of webrtcvad.Vad
            frames: a source of audio frames (sequence or generator)

        Returns:
            A generator that yields PCM audio data.

        """

        num_padding_frames = int(padding_duration_ms / frame_duration_ms)
        # We use a deque for our sliding window/ring buffer.
        ring_buffer = collections.deque(maxlen=num_padding_frames)
        # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
        # NOTTRIGGERED state.
        triggered = False

        voiced_frames = []
        for frame in frames:
            is_speech = vad.is_speech(frame.bytes, sample_rate)

            #sys.stdout.write('1' if is_speech else '0')
            if not triggered:
                ring_buffer.append((frame, is_speech))
                num_voiced = len([f for f, speech in ring_buffer if speech])
                # If we're NOTTRIGGERED and more than 90% of the frames in
                # the ring buffer are voiced frames, then enter the
                # TRIGGERED state.
                if num_voiced > 0.9 * ring_buffer.maxlen:
                    triggered = True
                    #sys.stdout.write('+(%s)' % (ring_buffer[0][0].timestamp,))
                    # We want to yield all the audio we see from now until
                    # we are NOTTRIGGERED, but we have to start with the
                    # audio that's already in the ring buffer.
                    for f, s in ring_buffer:
                        voiced_frames.append(f)
                    ring_buffer.clear()
            else:
                # We're in the TRIGGERED state, so collect the audio data
                # and add it to the ring buffer.
                voiced_frames.append(frame)
                ring_buffer.append((frame, is_speech))
                num_unvoiced = len([f for f, speech in ring_buffer if not speech])
                # If more than 90% of the frames in the ring buffer are
                # unvoiced, then enter NOTTRIGGERED and yield whatever
                # audio we've collected.
                if num_unvoiced > 0.9 * ring_buffer.maxlen:
                    #sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
                    triggered = False
                    yield b''.join([f.bytes for f in voiced_frames])
                    ring_buffer.clear()
                    voiced_frames = []
        if voiced_frames:
            yield b''.join([f.bytes for f in voiced_frames])

    def write_wave(self, path, audio, sample_rate):
        """
        Note:

        Args:
            path: string, a path for saving wav file
            audio: string, binary encoded wav
            sample_rate: integer, sampling rate

        Returns:

        """

        with contextlib.closing(wave.open(path, 'wb')) as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio)


class Frame(object):
    """
    Note:
        used in VAD class
        original note: Represents a "frame" of audio data.

    Attributes:
        __init__: constructs Frame class
    """

    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


if __name__=='__main__':

    #loop through subfolders of voxceleb (vox1dev,vox1test,vox2dev,vox2test)
    voxpath='ge2e_eng_data'
    sublist = os.listdir(voxpath)

    for i in sublist:
        wavpath = './ge2e_eng_data/{}/01_more_than_10utts_per_spkr/*/*/*'.format(i)
        save_dir = './ge2e_eng_data/{}/02_vad'.format(i)
        vad_mode = 1
        paths = glob.glob(wavpath)
        vad = VAD(vad_mode, save_dir)
        vad.run_vad_list(paths)
        print("done")
