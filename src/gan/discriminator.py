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


class Discriminator:
    """
    Base class for Discriminator/Critic.
    """
    def __init__(self, data_shape, features=1):
        self.data_shape = data_shape
        self.features = features
        self.discriminator = self._build_discriminator()

    def _build_discriminator(self):
        """
        Build the discriminator model.
        """
        raise NotImplementedError("Must define _build_discriminator method.")

    def load_discriminator(self, weights_path):
        """
        Load the discriminator weights.
        :param weights_path: path to discriminator weights
        """
        self.discriminator.load_weights(weights_path)

    def get_discriminator(self):
        """
        Get the discriminator model.
        """
        return self.discriminator

    def get_discriminator_summary(self):
        """
        Get the discriminator model summary.
        """
        return self.discriminator.summary()


class VanillaDiscriminator(Discriminator):
    """
    Vanilla Discriminator class. Uses Dense layers for binary classification.
    """
    def __init__(self, data_shape, features=1):
        super(VanillaDiscriminator, self).__init__(data_shape, features)

    def _build_discriminator(self):
        """
        Build the discriminator model.
        """
        discriminator = Sequential()
        discriminator.add(Dense(64, activation="relu", input_dim=self.data_shape))
        discriminator.add(Dense(128, activation="relu"))
        discriminator.add(Dense(1, activation="sigmoid"))
        discriminator.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        return discriminator


class LSTMDiscriminator(Discriminator):
    """
    LSTM Discriminator class. Uses LSTM layers for sequence data.
    """
    def __init__(self, data_shape, features=1):
        super(LSTMDiscriminator, self).__init__(data_shape, features)

    def _build_discriminator(self):
        """
        Build the discriminator model.
        """
        discriminator = Sequential()
        discriminator.add(LSTM(128, return_sequences=True, input_shape=(self.data_shape, self.features)))
        discriminator.add(LeakyReLU(alpha=0.2))
        discriminator.add(LSTM(256))
        discriminator.add(LeakyReLU(alpha=0.2))
        discriminator.add(Dense(1, activation="sigmoid"))
        discriminator.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        return discriminator


class WassersteinCritic(Discriminator):
    """
    Wasserstein Critic class. Returns a score instead of a binary classification.
    """
    def __init__(self, data_shape, features=1):
        super(WassersteinCritic, self).__init__(data_shape, features)

    def _build_discriminator(self):
        """
        Build the critic model.
        """
        critic = Sequential()
        #critic.add(LSTM(128, return_sequences=True, input_shape=(self.data_shape, self.features)))
        #critic.add(LeakyReLU(alpha=0.2))
        #critic.add(LSTM(256))
        #critic.add(LeakyReLU(alpha=0.2))
        critic.add(Dense(64, input_dim=self.data_shape))
        critic.add(LeakyReLU(alpha=0.2))
        critic.add(Dense(128))
        critic.add(LeakyReLU(alpha=0.2))
        critic.add(Dense(256))
        critic.add(LeakyReLU(alpha=0.2))
        critic.add(Dense(1, activation="linear"))
        return critic


if __name__ == "__main__":
    LATENT_DIM = 100
    DATA_SHAPE = 252
    vanilla_discriminator = VanillaDiscriminator(data_shape=DATA_SHAPE,
                                                 features=1)
    vanilla_discriminator.get_discriminator_summary()
