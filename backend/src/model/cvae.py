import jax
import jax.numpy as jnp
import haiku as hk
from typing import List



class VAE_Encoder(hk.Module):
    def __init__(self, hidden_dims: List = None, name='VAE_Encoder'):
        super().__init__(name=name)
        if hidden_dims is None:
            hidden_dims = [64, 64]
        self.modules = []
        for h_dim in hidden_dims:
            self.modules.append((hk.Conv2D(output_channels=h_dim,
                                           kernel_shape=3, stride=2, padding=[1, 1]), hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.999)))
        self.fc_mu = hk.Linear(2)
        self.fc_var = hk.Linear(2)

    def __call__(self, x):
        for layer_i, bn_i in self.modules:
            x = layer_i(x)
            x = bn_i(x, is_training=True)
            x = jax.nn.leaky_relu(x)
        x = hk.Flatten()(x)
        mean = self.fc_mu(x)
        log_std = self.fc_var(x)
        return [mean, log_std]


class VAE_Decoder(hk.Module):
    def __init__(self, hidden_dims: List = None, name='VAE_Decoder'):
        super().__init__(name=name)
        if hidden_dims is None:
            self.hidden_dims = [64, 64]
        self.modules = []
        self.conv_dim = 7
        for h_dim in self.hidden_dims:
            self.modules.append((hk.Conv2DTranspose(output_channels=h_dim,
                                                    kernel_shape=3, stride=2), hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.999)))
        self.finally_Linear = hk.Linear(1)
        self.decoder_input = hk.Linear(self.conv_dim*self.conv_dim*self.hidden_dims[-1])

    def __call__(self, x):
        x = self.decoder_input(x)
        x = x.reshape(-1, self.conv_dim, self.conv_dim, self.hidden_dims[-1])
        for layer_i, bn_i in self.modules:
            x = layer_i(x)
            x = bn_i(x, is_training=True)
            x = jax.nn.leaky_relu(x)
        x = self.finally_Linear(x)
        x = jax.nn.sigmoid(x)
        return x

def reparameterize(mean, stddev):
    z = mean + stddev * jax.random.normal(hk.next_rng_key(), mean.shape)
    return z

def vae(batch):
    x, y = batch[0], batch[1]
    mu, log_var = VAE_Encoder()(x)
    z = reparameterize(mu, log_var)
    y = jax.nn.one_hot(y,10)
    z = jnp.concatenate([z, y], axis=1)
    z = VAE_Decoder()(z)
    return [z, x, mu, log_var]
