import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm
import datetime
from datetime import datetime as dt
import gc

import sklearn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Conv2D, LeakyReLU, BatchNormalization, Dropout, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from tensorflow.keras.callbacks import ModelCheckpoint

from src.data.dataloader import *
from src.gan import *


def train():
    DATA_SHAPE = 252
    dataloader = YahooFinanceDataLoader(data_shape=DATA_SHAPE, returns=True, scaling_method="standardize")

    name = "gbm"
    LATENT_DIM = 100

    wgan = WGANGP(name=name,
                  latent_dim=LATENT_DIM,
                  data_shape=DATA_SHAPE,
                  lstm=False)

    EPOCHS = 5000
    BATCH_SIZE = 128
    EVAL_PERIOD = 100

    wgan.fit(dataloader,
             epochs=EPOCHS,
             eval_period=EVAL_PERIOD,
             batch_size=BATCH_SIZE,
             save_checkpoints=True)


if __name__ == "__main__":
    print("TensorFlow version: {}".format(tf.__version__))

    print("GPU Available: ", tf.test.is_gpu_available())
    print("Number of GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("GPUs: ", tf.config.list_physical_devices('GPU'))
    if tf.test.is_gpu_available():
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = None
    print("Execution Strategy: ", strategy)

    train()
