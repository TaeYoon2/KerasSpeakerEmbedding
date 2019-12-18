import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer
from tensorflow.python.ops import nn
from tensorflow.keras import backend as K


class Similarity_matrix(Layer):
    """
    Note:
        Calculate the similarites of utterances & centroids

    Attributes:
        __init__: set up the configurations of similarity matrix
        build: add up the variables of similarity matrix
        call: calculate the similarities in two ways; exclusive, inclusive
    """

    def __init__(self, num_speakers, num_utterance, is_exclusive, **kwargs):
        """
        Note:
            set up the configurations of similarity matrix;
        Args:
            num_speakers: the number of speakers in a batch
            num_utterance: the number of utterances per speaker in a batch
            is_exclusive: whether the self-embedding is included or not, when calculating the centroids

        Returns:

        """

        super(Similarity_matrix, self).__init__(**kwargs)
        self._num_speakers = num_speakers
        self._num_utterance = num_utterance
        self._is_exclusive = is_exclusive

    def build(self, input_shape):
        """
        Note:
            add up the variables; a weight & a bias
        Args:
            input_shape: the shape of input to be fed in call()
        Returns:

        """

        self.weight = self.add_weight("weight",shape=[1], dtype=tf.float32, initializer=tf.compat.v1.constant_initializer(value=[10], dtype=tf.float32),trainable=True)
        self.bias = self.add_weight("bias",shape=[1], dtype=tf.float32, initializer=tf.compat.v1.constant_initializer(value=[-5], dtype=tf.float32), trainable=True)

        super(Similarity_matrix, self).build(input_shape)

    def call(self, inputs):
        """
        Note:
            calculate the similarities; exclusive or inclusive
        Args:
            inputs:
                  inputs[0]: embeddings from utterances in a batch
                  inputs[1]: centroid embeddings per speaker in a batch
        Returns:
                  weighted_similarity
        """

        if self._is_exclusive:
            # [tot_utt, embed_dim]
            utterances = inputs[0]
            # [tot_utt, embed_dim, num_spkr]
            centroids = tf.keras.backend.permute_dimensions(inputs[1], [1, 0, 2])

            l2_utterances = nn.l2_normalize(utterances, axis=1)
            l2_centroids = nn.l2_normalize(centroids, axis=1)

            similarity = K.batch_dot(l2_utterances, l2_centroids, axes=[1, 1])
        else:
            l2_utterances = tf.nn.l2_normalize(inputs[0], axis=-1)
            l2_centroids = tf.nn.l2_normalize(tf.keras.backend.transpose(inputs[1]), axis=0)
            similarity = K.dot(l2_utterances, l2_centroids)

        self.weight = tf.clip_by_value(self.weight, 1e-6, np.infty)
        weighted_similarity = self.weight * similarity + self.bias

        return weighted_similarity

