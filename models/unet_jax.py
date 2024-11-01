import jax
from jax import numpy as jnp
from flax import linen as fnn

class Encoder(fnn.Module):
    features: int = 64
    training: bool = True

    @fnn.compact
    def __call__(self, x):
        z1 = fnn.Conv(self.features, kernel_size=(3, 3))(x)
        z1 = fnn.relu(z1)
        z1 = fnn.Conv(self.features, kernel_size=(3, 3))(z1)
        z1 = fnn.BatchNorm(use_running_average=not self.training)(z1)
        z1 = fnn.relu(z1)
        z1_pool = fnn.max_pool(z1, window_shape=(2, 2), strides=(2, 2))

        z2 = fnn.Conv(self.features * 2, kernel_size=(3, 3))(z1_pool)
        z2 = fnn.relu(z2)
        z2 = fnn.Conv(self.features * 2, kernel_size=(3, 3))(z2)
        z2 = fnn.BatchNorm(use_running_average=not self.training)(z2)
        z2 = fnn.relu(z2)
        z2_pool = fnn.max_pool(z2, window_shape=(2, 2), strides=(2, 2))

        z3 = fnn.Conv(self.features * 4, kernel_size=(3, 3))(z2_pool)
        z3 = fnn.relu(z3)
        z3 = fnn.Conv(self.features * 4, kernel_size=(3, 3))(z3)
        z3 = fnn.BatchNorm(use_running_average=not self.training)(z3)
        z3 = fnn.relu(z3)
        z3_pool = fnn.max_pool(z3, window_shape=(2, 2), strides=(2, 2))

        z4 = fnn.Conv(self.features * 8, kernel_size=(3, 3))(z3_pool)
        z4 = fnn.relu(z4)
        z4 = fnn.Conv(self.features * 8, kernel_size=(3, 3))(z4)
        z4 = fnn.BatchNorm(use_running_average=not self.training)(z4)
        z4 = fnn.relu(z4)
        z4_dropout = fnn.Dropout(0.5, deterministic=not self.training)(z4)
        z4_pool = fnn.max_pool(z4_dropout, window_shape=(2, 2), strides=(2, 2))

        z5 = fnn.Conv(self.features * 16, kernel_size=(3, 3))(z4_pool)
        z5 = fnn.relu(z5)
        z5 = fnn.Conv(self.features * 16, kernel_size=(3, 3))(z5)
        z5 = fnn.BatchNorm(use_running_average=not self.training)(z5)
        z5 = fnn.relu(z5)
        z5_dropout = fnn.Dropout(0.5, deterministic=not self.training)(z5)

        return z1, z2, z3, z4_dropout, z5_dropout


class Decoder(fnn.Module):
    features: int = 64
    training: bool = True

    @fnn.compact
    def __call__(self, z1, z2, z3, z4_dropout, z5_dropout):
        z6_up = jax.image.resize(z5_dropout, shape=(z5_dropout.shape[0], z5_dropout.shape[1] * 2, z5_dropout.shape[2] * 2, z5_dropout.shape[3]),
                                 method='nearest')
        z6 = fnn.Conv(self.features * 8, kernel_size=(2, 2))(z6_up)
        z6 = fnn.relu(z6)
        z6 = jnp.concatenate([z4_dropout, z6], axis=3)
        z6 = fnn.Conv(self.features * 8, kernel_size=(3, 3))(z6)
        z6 = fnn.relu(z6)
        z6 = fnn.Conv(self.features * 8, kernel_size=(3, 3))(z6)
        z6 = fnn.BatchNorm(use_running_average=not self.training)(z6)
        z6 = fnn.relu(z6)

        z7_up = jax.image.resize(z6, shape=(z6.shape[0], z6.shape[1] * 2, z6.shape[2] * 2, z6.shape[3]),
                                 method='nearest')
        z7 = fnn.Conv(self.features * 4, kernel_size=(2, 2))(z7_up)
        z7 = fnn.relu(z7)
        z7 = jnp.concatenate([z3, z7], axis=3)
        z7 = fnn.Conv(self.features * 4, kernel_size=(3, 3))(z7)
        z7 = fnn.relu(z7)
        z7 = fnn.Conv(self.features * 4, kernel_size=(3, 3))(z7)
        z7 = fnn.BatchNorm(use_running_average=not self.training)(z7)
        z7 = fnn.relu(z7)

        z8_up = jax.image.resize(z7, shape=(z7.shape[0], z7.shape[1] * 2, z7.shape[2] * 2, z7.shape[3]),
                                 method='nearest')
        z8 = fnn.Conv(self.features * 2, kernel_size=(2, 2))(z8_up)
        z8 = fnn.relu(z8)
        z8 = jnp.concatenate([z2, z8], axis=3)
        z8 = fnn.Conv(self.features * 2, kernel_size=(3, 3))(z8)
        z8 = fnn.relu(z8)
        z8 = fnn.Conv(self.features * 2, kernel_size=(3, 3))(z8)
        z8 = fnn.BatchNorm(use_running_average=not self.training)(z8)
        z8 = fnn.relu(z8)

        z9_up = jax.image.resize(z8, shape=(z8.shape[0], z8.shape[1] * 2, z8.shape[2] * 2, z8.shape[3]),
                                 method='nearest')
        z9 = fnn.Conv(self.features, kernel_size=(2, 2))(z9_up)
        z9 = fnn.relu(z9)
        z9 = jnp.concatenate([z1, z9], axis=3)
        z9 = fnn.Conv(self.features, kernel_size=(3, 3))(z9)
        z9 = fnn.relu(z9)
        z9 = fnn.Conv(self.features, kernel_size=(3, 3))(z9)
        z9 = fnn.BatchNorm(use_running_average=not self.training)(z9)
        z9 = fnn.relu(z9)

        y = fnn.Conv(1, kernel_size=(1, 1))(z9)
        #y = fnn.sigmoid(y)

        return y


class UNet(fnn.Module):
    features: int = 64

    @fnn.compact
    def __call__(self, x, train:bool=True):
        z1, z2, z3, z4_dropout, z5_dropout = Encoder(training=train)(x)
        y = Decoder(training=train)(z1, z2, z3, z4_dropout, z5_dropout)

        return y

