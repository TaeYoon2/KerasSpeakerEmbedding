import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.python.ops import nn


class Centroid_matrix(Layer):
    def __init__(self, num_speakers, num_utterance, **kwargs):
        super(Centroid_matrix, self).__init__(**kwargs)
        self._num_speakers = num_speakers
        self._num_utterance = num_utterance

    def build(self, input_shape):
        super(Centroid_matrix, self).build(input_shape)

    def call(self, inputs):

        # input shape [tot_utt, embed_dim]
        inputs = tf.keras.backend.permute_dimensions(inputs, [1, 0])  # [embed_dim, tot_utt]
        # centroid_column
        self_block = tf.keras.backend.ones(shape=[self._num_utterance, self._num_utterance], dtype=tf.float32) - tf.keras.backend.eye(self._num_utterance, dtype=tf.float32)
        self_block = self_block / (self._num_utterance - 1)
        # [num_spkr_utt, num_spkr_utt]
        centroid_block = tf.pad(self_block, [[0, 0], [0, (self._num_speakers - 1) * self._num_utterance]], name="normal_centroid_select_pad", constant_values=1/self._num_utterance)
        # [num_spkr_utt * num_spkr, num_spkr_utt]
        centroid_per_spkr = tf.pad(centroid_block, [[0, (self._num_speakers - 1) * self._num_utterance], [0, 0]], name="other_utterances_zero" , constant_values=0)
        # [tot_utt, tot_utt]
        #  ex) for spkr1
        # [[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,   ...   ]    {
        #  [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,   ...   ]       ~
        #  [1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1,   ...   ]
        #  [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1,   ...   ]
        #  [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1,   ...   ]
        #  [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,   ...   ]      10개의 동일 화자 어터런스 선택(그 중 자기 자신은 제외)하는 Linear Combination Matrix
        #  [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1,   ...   ]
        #  [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1,   ...   ]
        #  [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1,   ...   ]       ~
        #  [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,   ...   ]        }
        #  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   ...   ]
        #  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   ...   ]
        #  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   ...   ]
        #  [             ...                   ...   ]
        #  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   ...   ]
        # ]

        # [tot_utt, tot_utt]
        centroid_per_spkr_list = [tf.roll(centroid_per_spkr, axis=0, shift=spk_idx * self._num_utterance) for spk_idx in range(self._num_speakers)]
        # num_spkr * [tot_utt, tot_utt]
        centroid_list = tf.keras.backend.stack(centroid_per_spkr_list, axis=-1)
        # [tot_utt, tot_utt, num_spkr]

        self_exclusive_centroids = tf.keras.backend.dot(inputs, centroid_list)
        # [embed_dim, tot_utt] * [tot_utt, tot_utt, num_spkr]
        # ---> [embed_dim, tot_utt, num_spkr]
        return self_exclusive_centroids
