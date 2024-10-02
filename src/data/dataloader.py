import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm
import datetime
from datetime import datetime as dt
import sklearn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import yfinance as yf
from simulation import simulate_gbm


class DataLoader:
    """
    Base class for data loaders.
    """
    def __init__(self, data=None, data_shape=0, returns=False, scaling_method="standardize"):
        """
        Init method for DataLoader class.
        :param data: data source input
        :param data_shape: length of sequence
        :param returns: whether to use log returns or raw prices
        :param scaling_method: standardize or minmax
        """
        self.data_shape = data_shape
        self.returns = returns
        self.data = data
        self.raw_data = None
        self.scaler = None
        if scaling_method == "standardize":
            self.scaler = StandardScaler()
        elif scaling_method == "minmax":
            self.scaler = MinMaxScaler()

    def generate_data(self, n=100):
        """
        Generate data samples.
        :param n: number of samples to generate
        """
        raise NotImplementedError

    def generate_real_samples(self, n, lstm=False):
        """
        Generate real samples for GAN input
        :param n: number of samples to generate
        :param lstm: whether to reshape data for LSTM
        """
        X = self.generate_data(n)
        if lstm:
            X = X.reshape((n, self.data_shape, 1))
        else:
            X = X.reshape((n, self.data_shape))
        y = np.ones((n, 1))
        return X, y

    def reverse_preprocessing(self, data, initial_price=100):
        """
        Undo scaling and log returns to get raw prices.
        :param data: data to reverse preprocess
        :param initial_price: initial price to start from
        :return original spot price time series
        """
        data = data.reshape((data.shape[0], self.data_shape))
        # Reverse scaling
        log_returns = self.scaler.inverse_transform(data)
        returns = np.exp(log_returns)
        # Reverse log returns, recover the original price series
        initial_prices = initial_price * np.ones((data.shape[0], 1))
        original_prices = initial_prices * np.cumprod(returns, axis=1)
        return original_prices

    def get_raw_data(self):
        """
        Get raw data.
        :return: raw price data
        """
        return self.raw_data

    def get_raw_data_sample(self):
        """
        Get a random sample of raw data.
        :return: random sample of raw data
        """
        idx = np.random.randint(self.raw_data.shape[0] - self.data_shape)
        return self.raw_data[idx:idx + self.data_shape]


class GBMDataLoader(DataLoader):
    def __init__(self, data=None, data_shape=0, returns=False, scaling_method="standardize"):
        super().__init__(data, data_shape, returns, scaling_method)

    def generate_data(self, n=100):
        prices = simulate_gbm(n, N=self.data_shape+1 if self.returns else self.data_shape)
        self.raw_data = prices
        self.data = prices
        # preprocessing
        if self.returns:
            returns = self.data[:, 1:] / self.data[:, :-1]
            #returns = np.hstack([np.ones((self.data.shape[0], 1)), returns])  # keep returns length same as original price path
            self.data = np.log(returns)  # log returns
        if self.scaler is not None:
            self.data = self.scaler.fit_transform(self.data)
        return self.data

    def generate_real_samples(self, n, lstm=False):
        X = self.generate_data(n)
        if lstm:
            X = X.reshape((n, self.data_shape, 1))
        else:
            X = X.reshape((n, self.data_shape))
        y = np.ones((n, 1))
        return X, y


class YahooFinanceDataLoader(DataLoader):
    def __init__(self, data=None, data_shape=0, returns=False, scaling_method="standardize"):
        super().__init__(data, data_shape, returns, scaling_method)
        self._query_market_data()

    def _query_market_data(self):
        self.data = yf.download("NVDA", start="1962-07-01", end=dt.today())["Close"].to_numpy()
        self.raw_data = self.data
        # preprocessing
        if self.returns:
            self.data = np.log(self.data[1:] / self.data[:-1])  # log returns
        if self.scaler is not None:
            self.data = self.scaler.fit_transform(self.data.reshape(-1, 1))
        rolling_sequences = []
        for i in range(len(self.data) - self.data_shape):
            rolling_sequences.append(self.data[i:i + self.data_shape])
        self.data = np.array(rolling_sequences)

    def generate_data(self, n=100):
        idx = np.random.choice(self.data.shape[0], n, replace=False)
        return self.data[idx]


if __name__ == "__main__":
    # Test GBMDataLoader
    gbm_loader = GBMDataLoader(data_shape=252, returns=True, scaling_method="standardize")
    gbm_loader.generate_data(1000)

    # Test YahooFinanceDataLoader
    yahoo_loader = YahooFinanceDataLoader(data_shape=100, returns=True, scaling_method="standardize")
    data, _ = yahoo_loader.generate_data(1000)
    # one sample of data
    plt.plot(yahoo_loader.get_raw_data_sample())
    plt.show()
    # plot log returns
    for i in range(len(data)):
        plt.plot(data[i])
    plt.show()
    # plot full raw price data
    raw = yahoo_loader.get_raw_data()
    plt.plot(raw)
    plt.show()
    # reverse preprocessing and plot returns starting from initial price of 100
    reverse = yahoo_loader.reverse_preprocessing(data)
    for i in range(len(reverse)):
        plt.plot(reverse[i])
    plt.show()
    # plot one sample of reversed data
    plt.plot(reverse[0])
    plt.show()
