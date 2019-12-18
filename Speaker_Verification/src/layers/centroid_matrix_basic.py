import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.python.ops import nn

class Centroid_matrix_basic(Layer):
    """
    Note:
        Compute centroids of all speakers (including the utterance itself)

    Attributes:
        __init__:
        build: Creates the variables of the layer
        call: Compute centroid matrix of speaker embeddings
    """

    def __init__(self, num_speakers, num_utterance, **kwargs):
        """
        Note:

        Args:
            num_speakers: number of speakers (in the paper, it is 64)
            num_utterance: number of utterances per speaker (in the paper, it is 10)

        Returns:

        """

        super(Centroid_matrix_basic, self).__init__(**kwargs)
        self._num_speakers = num_speakers
        self._num_utterance = num_utterance

    def build(self, input_shape):
        """
        Note:
            Creates the variables of the layer according the input_shape (optional)

        Args:
            input_shape: [#total_utterance, #emb_dim]

        Returns:

        """

        super(Centroid_matrix_basic, self).build(input_shape)

    def call(self, inputs):
        """
        Note:
            Compute centroid matrix of speaker embeddings

        Args:
            inputs: the output of multi lstm and dense layer

        Returns:
            centroid: speaker centroid matrix [#spk, #emb_dim]
        """

        # Compute centroids of speaker embeddings
        centroid = tf.keras.backend.reshape(inputs, [self._num_speakers, self._num_utterance, -1])
        centroid = tf.keras.backend.mean(centroid, axis=1)

        return centroid
