from flax import linen as nn
import jax.numpy as jnp
from jax import jit, grad, lax, random
from jax.example_libraries import optimizers
from jax.example_libraries import stax
from jax.example_libraries.stax import Dense, FanOut, Relu, Softplus


def gaussian_sample(rng, mu, sigmasq):
  """Sample a diagonal Gaussian."""
  return mu + jnp.sqrt(sigmasq) * random.normal(rng, mu.shape)


class ImageEncoder(nn.Module):
    def __init__(self):
        _, self.encode = stax.serial(
            Dense(512), Relu,
            Dense(512), Relu,
            FanOut(2),
            stax.parallel(Dense(10), stax.serial(Dense(10), Softplus)),
        )

    def __call__(self, imgs):
        mu_z, sigmasq_z = self.encode(imgs)

        return mu_z, sigmasq_z


class ConditionEncoder(nn.Module):
    def __call__(self, y):
        out = nn.Embed(10, 10)(y)
        out = nn.relu(nn.Dense(10)(out))

        return out


class Decoder(nn.Module):
    def __call__(self, h_q):
        out = nn.relu(nn.Dense(512)(h_q))
        out = nn.relu(nn.Dense(512)(out))

        out = nn.Dense(28 * 28)(out)

        return out


class CVAE(nn.Module):
    def __init__(self):
        self.image_encoder = ImageEncoder()
        self.condition_encoder = ConditionEncoder()
        self.decoder = Decoder()

    def __call__(self, imgs, y):
        mu_z, sigmasq_z = self.image_encoder(imgs)
        cond = self.condition_encoder(y)

        out = jnp.concatenate([cond, gaussian_sample(rng=0, mu=mu_z, sigmasq=sigmasq_z)])

        out = self.decoder(out)
        
        return mu_z, sigmasq_z, out
