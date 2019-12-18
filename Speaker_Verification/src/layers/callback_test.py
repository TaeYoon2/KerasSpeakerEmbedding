import os
import tensorflow as tf
import pandas as pd
from tensorflow.python.keras import backend as K
from tensorflow.python.eager import context
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary import summary as tf_summary
from tensorflow.python.ops import summary_ops_v2
from sklearn.preprocessing import StandardScaler
from tensorflow.contrib.tensorboard.plugins import projector
from sklearn.decomposition import PCA
import numpy as np

# for eer
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
import sys
sys.path.append("../")
from utils.plot_2d import PlotEmb
import glob
import numpy as np
import re
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from scipy.optimize import brentq

class TESTCallback(tf.keras.callbacks.Callback):
    """
    Note:
        Callback function to observe the output result of the network

    Attributes:
        __init__: Constructs the class
        on_epoch_end: function called at the end of an epoch (during train mode)
        calculate_eer: calculate EER
        plot_emb_tsne: tsne plot
    """

    def __init__(self, eer_data, save_path, plot_spkr_num, period):
        """
        Note:

        Args:
            eer_data: test data for measuring eer
            save_path:
            plot_spkr_num: how many speakers to plot
            period: test tsne eer period

        Returns:

        """

        super(TESTCallback, self).__init__()
        self.batch_list = eer_data[0]
        self.lab_list = eer_data[1]
        self.save_path = save_path
        self.plot_spkr_num = plot_spkr_num
        self.period=period

    def on_epoch_end(self, epoch, logs={}):
        """
        Note:
            Called at the end of an epoch (during train mode)

        Args:
            epoch: int, index of epoch
            logs: dict results for training epoch

        Returns:

        """

        # Inference by dataset (self.xy or x_train)
        if np.mod(epoch, self.period) == 0:

            # make inferred embedding list
            inferred_emb_list =[]
            for one_batch in self.batch_list:
                inferred_batch = self.model.predict(one_batch, steps=1)
                inferred_emb_list.extend(inferred_batch[1])

            # get labels
            all_labels= []
            for batch_labels in self.lab_list:
                all_labels.extend(batch_labels)

            # make tsne plot
            self.plot_emb_tsne(inferred_emb_list, all_labels, epoch, self.plot_spkr_num)

            # calculate eer and print eer
            eer=self.calculate_eer(np.asarray(inferred_emb_list), all_labels)
            print(f"EER @ {epoch} : {eer}% ")

    def calculate_eer(self, inferred_emb_list, all_labels):
        """
        Note:
            Calculate EER
        Args:
            inferred_emb_list: inferred embedding list
            all_labels: original labels of utterances

        Returns:
            eer: error rate

        """

        # score (1920, 1920) and label
        score = np.dot(inferred_emb_list, inferred_emb_list.T)
        label = np.zeros(score.shape)

        # patterns to find vox and lib data
        pattern_vox = re.compile(r"(?P<spkr_vox>id\d{5})_(?P<utt_vox>(\S{11})_(\d{5}))\.npy")
        pattern_lib = re.compile(r"(?P<spkr_lib>\d+)-(?P<utt_lib>(\d+)-(\d+))\.npy")

        # fill spkr_label
        spkr_label = []
        for uttid in all_labels:
            m_vox=re.search(pattern_vox,uttid)
            m_lib=re.search(pattern_lib,uttid)
            if m_vox:
                spkr_label.append("VOX"+m_vox.group("spkr_vox"))
            elif m_lib:
                spkr_label.append("LIB"+m_lib.group("spkr_lib"))

        # score and label
        for i in range(label.shape[0]):
            idx = tuple([n for n,x in enumerate(spkr_label) if x==spkr_label[i]])
            label[i,idx]+=1
        label = label[~np.eye(label.shape[0],dtype=bool)]
        score = score[~np.eye(score.shape[0],dtype=bool)]

        # compute eer
        fpr, tpr, thresholds = roc_curve(label, score)
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        return eer

    def plot_emb_tsne(self, inferred_emb_list, all_labels, num_epoch, plot_spkr_num):
        """
        Note:
            tsne plot

        Args:
            inferred_emb_list: inferred embedding list
            all_labels: original labels of utterances
            num_epoch: num_epoch for tsne
            plot_spkr_num: how many speakers to plot

        Returns:

        """

        # tsne plot
        plotemb=PlotEmb(inferred_emb_list,all_labels,num_epoch)
        plotemb.plot_tsne(plot_spkr_num)
        return print(f"epoch {num_epoch} tsne plot saved to test data dir")
