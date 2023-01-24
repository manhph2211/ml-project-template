import jax.numpy as jnp
from jax import random


def gaussian_kl(mu, sigmasq):
  """KL divergence from a diagonal Gaussian to the standard Gaussian."""
  return -0.5 * jnp.sum(1. + jnp.log(sigmasq) - mu**2. - sigmasq)

def bernoulli_logpdf(logits, x):
  """Bernoulli log pdf of data x given logits."""
  return -jnp.sum(jnp.logaddexp(0., jnp.where(x, -1., 1.) * logits))

def elbo(logits_x, images, mu_z, sigmasq_z):
  """Monte Carlo estimate of the negative evidence lower bound."""

  return bernoulli_logpdf(logits_x, images) - gaussian_kl(mu_z, sigmasq_z)