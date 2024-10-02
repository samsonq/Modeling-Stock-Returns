import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


def simulate_gbm(n, S0=100, mu=0.05, sigma=0.2, T=1, N=252):
    """
    Simulate n paths of a geometric Brownian motion.
    :param n: number of samples to simulate
    :param S0: initial stock price
    :param mu: expected return (drift)
    :param sigma: volatility (standard deviation)
    :param T: time period (e.g., 1 year)
    :param N: number of time steps (e.g., 252 trading days in a year)
    :return: simulated GBM paths
    """
    dt = T / N  # Time step size
    all_paths = np.zeros((n, N))
    for i in range(n):
        # Simulate stock prices
        S = np.zeros(N)
        S[0] = S0
        for t in range(1, N):
            Z_t = np.random.normal(0, 1)  # Generate a random standard normal variable
            S[t] = S[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z_t)
        all_paths[i] = S
    return all_paths


def simulate_gbm_paths(n):
    """
    Plot n simulation paths of a geometric Brownian motion.
    :param n: number of samples to simulate
    """
    paths = simulate_gbm(n)
    print(paths.shape)
    for i in paths:
        plt.plot(i)
    plt.show()


if __name__ == "__main__":
    simulate_gbm_paths(100)
