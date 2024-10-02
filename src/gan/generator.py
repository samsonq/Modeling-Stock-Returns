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


class Generator:
    """
    Base class for Generator.
    """
    def __init__(self, latent_dim, data_shape, features=1):
        self.latent_dim = latent_dim
        self.data_shape = data_shape
        self.features = features
        self.generator = self._build_generator()

    def _build_generator(self):
        """
        Build the generator model.
        """
        raise NotImplementedError("Must define _build_generator method.")

    def load_generator(self, weights_path):
        """
        Load the generator weights.
        :param weights_path: path to generator weights
        """
        self.generator.load_weights(weights_path)

    def get_generator(self):
        """
        Get the generator model.
        """
        return self.generator

    def get_generator_summary(self):
        """
        Get the generator model summary.
        """
        return self.generator.summary()


class VanillaGenerator(Generator):
    """
    Vanilla Generator class. Dense layers to generate synthetic sequential data.
    """
    def __init__(self, latent_dim, data_shape, features=1):
        super(VanillaGenerator, self).__init__(latent_dim, data_shape, features)

    def _build_generator(self):
        """
        Build the generator model.
        """
        generator = Sequential()
        generator.add(Dense(64, activation="relu", input_dim=self.latent_dim))
        generator.add(Dense(128, activation="relu"))
        generator.add(Dense(self.data_shape, activation="linear"))
        return generator


class LSTMGenerator(Generator):
    """
    LSTM Generator class. LSTM layers to generate synthetic sequential data.
    """
    def __init__(self, latent_dim, data_shape, features=1):
        super(LSTMGenerator, self).__init__(latent_dim, data_shape, features)

    def _build_generator(self):
        """
        Build the generator model.
        """
        generator = Sequential()
        generator.add(LSTM(128, return_sequences=True, input_shape=(self.latent_dim, 1)))
        generator.add(LeakyReLU(alpha=0.2))
        generator.add(LSTM(256))
        generator.add(LeakyReLU(alpha=0.2))
        generator.add(Dense(self.data_shape, activation="linear"))
        generator.add(Reshape((self.data_shape, self.features)))
        return generator


class WassersteinGenerator(Generator):
    """
    Wasserstein Generator class.
    """
    def __init__(self, latent_dim, data_shape, features=1):
        super(WassersteinGenerator, self).__init__(latent_dim, data_shape, features)

    def _build_generator(self):
        """
        Build the generator model.
        """
        generator = Sequential()
        #generator.add(LSTM(128, return_sequences=True, input_shape=(self.latent_dim, 1)))
        #generator.add(LeakyReLU(alpha=0.2))
        #generator.add(LSTM(256))
        #generator.add(LeakyReLU(alpha=0.2))
        #generator.add(Dense(self.data_shape, activation="linear"))
        #generator.add(Reshape((self.data_shape, self.features)))
        generator.add(Dense(64, input_dim=self.latent_dim))
        generator.add(LeakyReLU(alpha=0.2))
        generator.add(Dense(128))
        generator.add(LeakyReLU(alpha=0.2))
        generator.add(Dense(256))
        generator.add(LeakyReLU(alpha=0.2))
        generator.add(Dense(self.data_shape, activation="linear"))
        return generator


def generate_noise(latent_dim, n):
    """
    Generate noise for the generator.
    :param latent_dim: latent dimension
    :param n: number of samples to generate
    :return: noise input
    """
    noise = np.random.randn(latent_dim*n)
    return noise.reshape(n, latent_dim)


if __name__ == "__main__":
    LATENT_DIM = 100
    DATA_SHAPE = 252
    vanilla_generator = VanillaGenerator(latent_dim=LATENT_DIM,
                                         data_shape=DATA_SHAPE,
                                         features=1)
    vanilla_generator.get_generator_summary()
