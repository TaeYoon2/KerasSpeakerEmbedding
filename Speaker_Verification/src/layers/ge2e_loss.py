import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer
from tensorflow.python.ops import nn


class Ge2e_loss(Layer):
    """
    Note:
        Compute the loss of ge2e in two ways; softmax, contrast

    Attributes:
        __init__: constructs Ge2e_loss class
        call: compute the ge2e loss
    """

    def __init__(self, num_speakers, num_utterance, loss_type="softmax", **kwargs):
        """
        Note:
            set up the loss configurations; No. speakers, No. utterances, loss_type
        Args:
            num_speakers: the number of speakers
            num_utterance: the number of utterances
            loss_type: "softmax" or "contrast"

        Returns:

        """

        self._num_speakers = num_speakers
        self._num_utterance = num_utterance
        self._loss_type = loss_type
        super(Ge2e_loss, self).__init__(**kwargs)


    def call(self, inputs):
        """
        Note:
            compute the ge2e loss of a batch
        Args:
            inputs: the similarities between batch utterances & batch centroids
                    [640, 64]
        Returns:
            loss: ge2e loss
        """

        if self._loss_type == "softmax":
            softmax_similarities = -(inputs - tf.math.log(tf.reduce_sum(tf.exp(inputs), axis=-1, keepdims=True)+ 1e-6))
            softmax_losses = []
            index = 0
            for j in range(self._num_speakers):
                for i in range(self._num_utterance):
                    softmax_losses.append(softmax_similarities[index,:][j])
                    index += 1
            loss = tf.keras.backend.sum(softmax_losses, keepdims=True)

        elif self._loss_type == "contrast":
            raise NotImplementedError
            # contrast loss : - positive + max(negatives)
            loss_positive = tf.math.sigmoid(loss_positive)

            self_block = tf.keras.backend.zeros(shape=[self._num_utterance, self._num_utterance], dtype=tf.float32)
            neg_blocks = tf.keras.backend.ones(shape=[self._num_utterance, (self._num_speakers - 1) * self._num_utterance], dtype=tf.float32)
            # [num_spkr_utt, tot_utt]
            mask_per_spkr = tf.keras.backend.concatenate([self_block, neg_blocks], axis=1)
            mask_per_spkr_list = [tf.roll(mask_per_spkr, axis=1, shift=spk_idx * self._num_utterance) for spk_idx in range(self._num_speakers)]
            neg_mask = tf.keras.backend.concatenate(mask_per_spkr_list, axis=0)

            loss_negative = tf.keras.backend.max(tf.math.sigmoid(neg_mask * inputs), axis=1, keepdims=True)
            #[tot_utt, 1]
            loss = 1  - loss_positive + loss_negative
            loss = tf.keras.backend.sum(loss)


        return loss

