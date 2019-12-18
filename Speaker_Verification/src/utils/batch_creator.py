import os
import sys
import time
import glob

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import json
import random
import numpy as np
import librosa
from multiprocessing import Pool

### USER DEFINED CLASS
from random_selector import RandomSelector
from mel_splitter import MelSplitter

class BatchCreator(object):
    """
    Note:
        random selects files & random split into a batch

    Attributes:
        __init__: constructs BatchCreator class

    """

    def __init__(self, parameters):
        """
        Note:

        Args:
            parameters: parameters for RandomSelector & MelSplitter

        Returns:

        """

        self._parameters = parameters
        self._random_selector = RandomSelector(parameters)
        self._mel_splitter = MelSplitter(parameters)
        return

    def get_batch_mel_path(self):
        """
        Note:

        Args:

        Returns:
            batch_mel_paths_list: list, randomely selected npy file paths by RandomSelector

        """

        batch_mel_paths_list = self._random_selector.random_batch_select()
        return batch_mel_paths_list

    #multiprocess 처리
    def multi_load_mels(self,batch_mel_paths_list,pool):
        """
        Note:
            uses Pool object for faster loading of multiple npy files

        Args:
            batch_mel_paths_list: list, containing path of npy files
            pool: created pool object

        Returns:
            datadict: dictionary, containing multiple outputs from load_single_mel_npy function

        """

        results = pool.map(self.load_single_mel_npy, batch_mel_paths_list)
        pool.close()
        pool.join()

        datadict=dict()
        datadict['mel'] = [i[0] for i in results]
        datadict['mel_len'] = [i[1] for i in results]
        datadict['spk_list'] = [i[2] for i in results]

        return datadict

    def load_single_mel_npy(self,batch_mel_path):
        """
        Note:
            load saved mel npy file

        Args:
            batch_mel_path: string, a path that indicates saved .npy file

        Returns:
            mel: np array, loaded mel array
            len(mel): integer, length of the loaded mel array
            file_name: string, basename of the input path

        """

        mel = np.load(batch_mel_path)
        file_name = os.path.basename(batch_mel_path)

        return mel, len(mel), file_name

    def create_batch(self,datadict):
        """
        Note:
            cuts loaded mel arrays into one batch using MelSplitter

        Args:
            datadict: dictionary, output of get_batch_mel_path function containing mel, mel_len, file_name

        Returns:
            one_batch: np array, a splitted batch

        """

        selected_frames = self._mel_splitter.get_seg_assignment(datadict,rand_seg=self._parameters['random_segment'])
        sliced_mels = self._mel_splitter.split_all_mels(selected_frames, datadict)
        one_batch = np.asarray(sliced_mels)
        return one_batch

