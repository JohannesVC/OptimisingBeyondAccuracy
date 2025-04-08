import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from .data_loader import *
from .metrics import *
from .utils import *
from .subclasses import *
from .bi_encoder import OptimisedModel
from . import subclasses
from pathlib import Path

subclasses_list = [name for name in dir(subclasses) 
                   if not name.startswith("_")]

data_folder = Path('../.data/')

import keras; keras.utils.set_random_seed(151)
from keras.api import layers, ops
from keras_hub import layers as nlp_layers

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import seaborn as sns

sns.set_theme(style='dark')
plt.rcParams['text.usetex'] = True

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    for gpu in gpus:
        print("Available device:", gpu.device_type, '\n')
else: print('GPU not detected.')

__all__ = [
    'DataLoader',
    'OptimisedModel',
    'log_results',
    'get_metrics',
    'get_norm_metrics',
    'intra_list_diversity',
    'data_folder',
    'keras', 'ops',
    'layers',
    'nlp_layers',
    'np', 'pd', 'plt', 'sns'
    ] + subclasses_list

