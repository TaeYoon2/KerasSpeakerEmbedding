import tensorflow as tf
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras import backend as K
import numpy as np
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD

class Ge2eOptimizer(SGD):
    """
    Note:
        Inherited from tf.keras.optimizer.SGD

    Attributes:
        __init__: constructs Ge2eOptimizer class
        get_updates: compute and modify gradients
    """

    def __init__(self,
             learning_rate=0.01,
             momentum=0.0,
             nesterov=False,
             **kwargs):
        """
        Note:

        Args:
            learning_rate: SGD parameter
            momentum: SGD parameter
            nesterov: SGD parameter

        Returns:

        """

        super(Ge2eOptimizer, self).__init__(**kwargs)

    def get_updates(self, loss, params):
        """
        Note:
            Compute gradients and modify them according to the GE2E paper

        Args:
            loss: loss to be minimized
            params: all trainable variables in the model

        Returns:
            A list of modified gradients to be applied
        """

        # Compute gradients of variables by layer group (AddN: LSTM, desne: projection, similarity: similarity matrix)
        grads = self.get_gradients(loss, params)
        lstm_grads = list(filter(lambda x: "AddN" in x.name, grads))
        dense_grads = list(filter(lambda x: "dense" in x.name, grads))
        sim_grads = list(filter(lambda x: "similarity" in x.name, grads))

        # Modify the gradients
        lstm_grads_processed = [tf.clip_by_norm(g, 3) for g in lstm_grads]
        dense_grads_processed = [g*0.5 for g in dense_grads]
        sim_grads_processed = [g*0.01 for g in sim_grads]

        # Concatenate the modified gradients into a list by the original variable order in the model
        all_grads_processed = lstm_grads_processed[:3] + dense_grads_processed[:2] + lstm_grads_processed[3:6]+ dense_grads_processed[2:4] + lstm_grads_processed[6:] + dense_grads_processed[4:] + sim_grads_processed

        # Zip the modified gradient list and the variable list
        grads_and_vars = list(zip(all_grads_processed, params))

        return [self.apply_gradients(grads_and_vars)]
