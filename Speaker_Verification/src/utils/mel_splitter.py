import numpy as np
import os
import random
import glob


class MelSplitter(object):
    """
    Note:
        Split mel array into a batch with random length and random starting point

    Attributes:
        __init__: constructs MelSplitter class
        get_seg_assignment: assign start and end frame index for each mel input
        split_mel: split mel based on the output of get_seg_assignment
        split_all_mels: same as split_mel, looping over multiple mel arrays from datadict
    """

    def __init__(self, config):
        """
        Note:

        Args:
            min_frames: integer, minimum length of split batch (frame)
            max_frames: integer, maximum length of split batch (frame)
            hop_size: integer, fft hop size (msec)
            win_len: integer, fft window size (msec)
            rand_seg: boolean, if True, assign length and starting frame randomly

        Returns:

        """

        self.min_frames = config["min_frames"] #140
        self.max_frames = config["max_frames"] #180
        self.hop_size = config["fft_hop_size"] #10ms
        self.win_len = config["fft_window_size"] #25ms
        self.rand_seg = config["random_segment"]
        self.seg_len = config["uniform_seg_frames"] #140ms; used only when RANDOM_SEGMENT=1


    def get_seg_assignment(self,datadict,rand_seg):

        """
        Note:
            assign start and end frame index for each mel input

        Args:
            datadict: dictionary, 'mel' is a list of np arrays and 'mel_len' is a list of len(mel[i])
            rand_seg: boolean, if True, assign length and starting frame randomly

        Returns:
            start_end_frame_for_each_mel: list of tuples, share index with datadict['mel']

        """

        print(f'seg random {rand_seg}')
        if rand_seg:
            shortest_idx = np.argmin(datadict['mel_len'])
            shortest_mel_len = datadict['mel_len'][shortest_idx]
            print(f"shortest_idx {shortest_idx} shortest_mel_len {shortest_mel_len}")
            #add random len extra frame [140,141, ... ,179,180]
            max_extra = shortest_mel_len - self.min_frames #13
            print(f"max_extra {max_extra}" )
            extra_frame = random.randint(0, max_extra)
            batch_frame_len = self.min_frames + extra_frame
            all_mel_len = datadict['mel_len']
            start_frame_for_each_mel = [random.randint(0, i-batch_frame_len) for i in all_mel_len]
            start_end_frame_for_each_mel = [(i, i+batch_frame_len) for i in start_frame_for_each_mel]

        else: #rand_seg=False
            batch_frame_len = self.seg_len

            all_mel_len = datadict['mel_len']
            start_frame_for_each_mel = [random.randint(0, i-batch_frame_len) for i in all_mel_len]
            start_end_frame_for_each_mel = [(i, i+batch_frame_len) for i in start_frame_for_each_mel]

        return start_end_frame_for_each_mel

    def split_mel(self, mel, start_end_frame):
        """
        Note:
            split mel based on the output of get_seg_assignment

        Args:
            mel: np array, mel feature
            start_end_frame: tuple, containing two integers which indicates start and end frame index individually

        Returns:
            mel_segment: np array, splitted output

        """

        mel_segment = mel[start_end_frame[0]:start_end_frame[1]]

        return mel_segment


    def split_all_mels(self, start_end_frame_for_each_mel, datadict):
        """
        Note:
            same as split_mel, looping over multiple mel arrays from datadict
        Args:
            start_end_frame_for_each_mel:
            datadict: dictionary, mel is a list of np arrays and mel_len is a list

        Returns:
            sliced_mels: a LIST of batches (num batches differ by inital wav selection)

        """

        mel_list = datadict['mel']
        sliced_mels = []
        for i, start_end_frame in enumerate(start_end_frame_for_each_mel):
            mel_segment = self.split_mel(mel_list[i], start_end_frame)
            sliced_mels.append(mel_segment)
        return sliced_mels
