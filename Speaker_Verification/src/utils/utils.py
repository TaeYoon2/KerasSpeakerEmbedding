import numpy as np
import os
import glob
from utils.infer_process import InferPrep


def load_batch(filepath):
    batch=np.load(filepath)
    return batch,filepath

def lr_scheduler(epoch):
	init_lr = 0.01
	drop_step = 30000000
	drop_rate = 0.5
	decay = drop_rate ** ((epoch * self._steps_per_epoch[0]) // drop_step)
	return init_lr * decay