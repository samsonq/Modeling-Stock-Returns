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
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from tensorflow.keras.callbacks import ModelCheckpoint

from generator import *
from discriminator import *


class GAN:
    """
    Base class for GAN.
    """
    def __init__(self, latent_dim, data_shape, learning_rate=0.00005, lstm=False, name="default"):
        self.name = name
        self.latent_dim = latent_dim
        self.data_shape = data_shape
        self.lstm = lstm
        self.learning_rate = learning_rate
        self.generator = None
        self.discriminator = None
        self.gan = None
        self.d_losses = np.array([])
        self.g_losses = np.array([])
        self._build_gan()

    def _build_gan(self):
        raise NotImplementedError("Must define _build_gan method in subclass.")

    def _summarize_performance(self, epoch, dataloader, n=100):
        raise NotImplementedError("Must define _summarize_performance method in subclass.")

    def generate_noise(self, n):
        noise = np.random.randn(self.latent_dim * n)
        if self.lstm:
            return noise.reshape(n, self.latent_dim, 1)
        return noise.reshape(n, self.latent_dim)

    def generate_data(self, n):
        noise = self.generate_noise(n)
        # generator predict
        synthetic_data = self.generator.predict(noise)
        return synthetic_data

    def generate_fake_samples(self, n):
        return self.generate_data(n), np.zeros((n, 1))

    def load_generator(self, weights_path):
        self.generator.load_weights(weights_path)

    def load_discriminator(self, weights_path):
        self.discriminator.load_weights(weights_path)

    def load_gan(self, weights_path):
        self.gan.load_weights(weights_path)

    def load_weights(self, generator_weights, discriminator_weights, gan_weights):
        self.load_generator(generator_weights)
        self.load_discriminator(discriminator_weights)
        self.load_gan(gan_weights)

    def save_weights(self, path="/weights"):
        if not os.path.exists(path):
            os.makedirs(path)
        self.generator.save_weights(os.path.join(path, "{}_generator_weights.h5".format(self.name)))
        self.discriminator.save_weights(os.path.join(path, "{}_discriminator_weights.h5".format(self.name)))
        if self.gan is not None:
            self.gan.save_weights(os.path.join(path, "{}_gan_weights.h5".format(self.name)))

    def get_generator(self):
        return self.generator

    def get_generator_summary(self):
        print("Generator Summary:")
        print(self.generator.summary())

    def get_discriminator(self):
        return self.discriminator

    def get_discriminator_summary(self):
        print("Discriminator Summary:")
        print(self.discriminator.summary())

    def get_gan(self):
        return self.gan

    def get_gan_summary(self):
        print("GAN Summary:")
        print(self.gan.summary())


class VanillaGAN(GAN):
    """
    Vanilla GAN. Dense layers to generate data.
    """
    def __init__(self, latent_dim, data_shape, learning_rate=0.00005, lstm=False, name="default"):
        super(VanillaGAN, self).__init__(latent_dim, data_shape, learning_rate, lstm, name)
        self.generator = VanillaGenerator(latent_dim, self.data_shape).get_generator()
        self.discriminator = VanillaDiscriminator(self.data_shape).get_discriminator()

    def _build_gan(self):
        self.discriminator.trainable = False
        self.gan = Sequential()
        self.gan.add(self.generator)
        self.gan.add(self.discriminator)
        self.gan.compile(loss='binary_crossentropy', optimizer='adam')

    def fit(self, dataloader, epochs, batch_size, eval_period=100, save_checkpoints=True):
        half_batch = int(batch_size / 2)
        for epoch in tqdm(range(epochs)):
            # generator fake data
            x_fake, y_fake = self.generate_fake_samples(n=half_batch)
            x_real, y_real = dataloader.generate_real_samples(n=half_batch)
            d_loss_real = self.discriminator.train_on_batch(x_real, y_real)
            d_loss_fake = self.discriminator.train_on_batch(x_fake, y_fake)
            d_loss = 0.5 * np.add(d_loss_real[0], d_loss_fake[0])
            self.d_losses = np.append(self.d_losses, d_loss)

            x_gan, y_gan = self.generate_noise(batch_size), np.ones((batch_size, 1))
            g_loss = self.gan.train_on_batch(x_gan, y_gan)
            self.g_losses = np.append(self.g_losses, g_loss)

            if (epoch + 1) % eval_period == 0:
                self._summarize_performance(epoch + 1, dataloader)
                if save_checkpoints:
                    # self.generator_checkpoint.on_epoch_end(epoch, g_loss)
                    # self.discriminator_checkpoint.on_epoch_end(epoch, d_loss)
                    # self.gan_checkpoint.on_epoch_end(epoch)
                    self._save_checkpoints(epoch + 1)
            # gc.collect()

    def _summarize_performance(self, epoch, dataloader, n=100):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 9))
        # losses
        print(f"{epoch} [D loss: {self.d_losses[-1]:.4f}] [G loss: {self.g_losses[-1]:.4f}]")
        ax1.plot(self.g_losses, color="red")
        ax1.plot(self.d_losses, color="blue")
        ax1.legend(["Generator Loss", "Discriminator Loss"])
        # generator performance
        x_real, y_real = dataloader.generate_real_samples(n)
        _, real_acc = self.discriminator.evaluate(x_real, y_real, verbose=0)
        x_fake, y_fake = self.generate_fake_samples(n)
        _, fake_acc = self.discriminator.evaluate(x_fake, y_fake, verbose=0)
        print("Epoch: {}".format(epoch))
        print("Real Accuracy: {}; Fake Accuracy: {}".format(real_acc, fake_acc))
        # ax2.plot(x_real[0], color="red")
        # ax2.plot(x_fake[0], color="blue")
        # ax2.legend(["Real Data", "Fake Data"])
        original_prices = dataloader.reverse_preprocessing(x_real)
        for i in range(len(original_prices)):
            ax2.plot(original_prices[i])
        fake_prices = dataloader.reverse_preprocessing(x_fake)
        for i in range(len(fake_prices)):
            ax3.plot(fake_prices[i])
        plt.show()


class WGANGP(GAN):
    """
    Wasserstein GAN with Gradient Penalty.
    """
    def __init__(self, latent_dim, data_shape, learning_rate=0.00005, lstm=False, name="default"):
        super(WGANGP, self).__init__(latent_dim, data_shape, learning_rate, lstm, name)
        self.generator = WassersteinGenerator(latent_dim, self.data_shape).get_generator()
        self.discriminator = WassersteinCritic(self.data_shape).get_discriminator()

    def _build_gan(self):
        self.generator_optimizer = RMSprop(learning_rate=self.learning_rate)
        self.critic_optimizer = RMSprop(learning_rate=self.learning_rate)
        self.generator.compile(loss=self.wasserstein_loss, optimizer=self.generator_optimizer)
        self.discriminator.compile(loss=self.wasserstein_loss, optimizer=self.critic_optimizer)

    @staticmethod
    def wasserstein_loss(y_true, y_pred):
        return tf.reduce_mean(y_true * y_pred)

    def gradient_penalty(self, real_samples, fake_samples):
        # alpha = tf.random.uniform(shape=[real_samples.shape[0], 1], minval=0., maxval=1.)
        # interpolated = alpha * real_samples + (1 - alpha) * fake_samples
        # with tf.GradientTape() as tape:
        #    tape.watch(interpolated)
        #    d_interpolated = self.critic(interpolated, training=True)
        # gradients = tape.gradient(d_interpolated, [interpolated])[0]
        # gradient_penalty = tf.reduce_mean(tf.square(tf.norm(gradients, axis=1) - 1))
        # return gradient_penalty
        # Create the interpolated image
        # if self.lstm:
        alpha = tf.random.uniform([real_samples.shape[0], 1], 0.0, 1.0)
        # else:
        #     alpha = tf.random.uniform([real_samples.shape[0], 1], 0.0, 1.0)
        interpolated = real_samples + alpha * (fake_samples - real_samples)
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.discriminator(interpolated, training=True)
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[0, 1]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def fit(self, dataloader, epochs, batch_size, critic_iterations=5, eval_period=100, save_checkpoints=True):
        # real_labels = -np.ones((batch_size, 1))  # -1 for real samples in WGAN
        # fake_labels = np.ones((batch_size, 1))  # 1 for fake samples in WGAN
        half_batch = int(batch_size / 2)
        for epoch in tqdm(range(epochs)):
            c_losses = np.array([])
            for _ in range(critic_iterations):
                x_fake, _ = self.generate_fake_samples(n=half_batch)
                x_real, _ = dataloader.generate_real_samples(n=half_batch)
                # Train the critic
                with tf.GradientTape() as disc_tape:
                    real_preds = self.discriminator(x_real, training=True)
                    fake_preds = self.discriminator(x_fake, training=True)
                    gp = self.gradient_penalty(x_real, x_fake)
                    c_loss = tf.reduce_mean(fake_preds) - tf.reduce_mean(real_preds) + 10 * gp
                    c_losses = np.append(c_losses, c_loss)
                gradients = disc_tape.gradient(c_loss, self.discriminator.trainable_variables)
                self.critic_optimizer.apply_gradients(zip(gradients, self.discriminator.trainable_variables))
            self.d_losses = np.append(self.d_losses, np.mean(c_losses))
            noise = self.generate_noise(batch_size)
            with tf.GradientTape() as gen_tape:
                fake_data = self.generator(noise, training=True)
                fake_preds = self.discriminator(fake_data, training=True)
                g_loss = -tf.reduce_mean(fake_preds)
                self.g_losses = np.append(self.g_losses, g_loss)

            gradients = gen_tape.gradient(g_loss, self.generator.trainable_variables)
            self.generator_optimizer.apply_gradients(zip(gradients, self.generator.trainable_variables))

            if (epoch + 1) % eval_period == 0:
                self._summarize_performance(epoch + 1, dataloader)

    def _summarize_performance(self, epoch, dataloader, n=100):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 9))
        # losses
        print(f"{epoch} [C loss: {self.c_losses[-1]:.4f}] [G loss: {self.g_losses[-1]:.4f}]")
        ax1.plot(self.g_losses, color="red")
        ax1.plot(self.c_losses, color="blue")
        ax1.legend(["Generator Loss", "Critic Loss"])
        ax1.title("Generator vs. Critic Loss over Epochs")
        # generator performance
        x_real, _ = dataloader.generate_real_samples(n)
        x_fake, _ = self.generate_fake_samples(n)
        original_prices = dataloader.reverse_preprocessing(x_real)
        for i in range(len(original_prices)):
            ax2.plot(original_prices[i])
        ax2.title("Real Market Data")
        fake_prices = dataloader.reverse_preprocessing(x_fake)
        for i in range(len(fake_prices)):
            ax3.plot(fake_prices[i])
        ax3.title("GAN-Generated Market Data")
        plt.savefig("Epoch_{}_Progress.png".format(epoch))
        plt.show()


if __name__ == "__main__":
    name = "gbm"
    LATENT_DIM = 100
    DATA_SHAPE = 252

    wgan = WGANGP(name=name,
                  latent_dim=LATENT_DIM,
                  data_shape=DATA_SHAPE,
                  lstm=False)
