import sys
sys.path.append(".")
import jax.numpy as jnp
import pickle
import matplotlib.pyplot as plt
from src.model.cvae import vae
import os
import haiku as hk
import numpy as np
import jax

vae = hk.transform_with_state(vae)
dummy_x = jnp.array(np.random.rand(8, 28, 28, 2))
dummy_y = jnp.array(np.random.rand(8))
rng_key = jax.random.PRNGKey(42)
params, state = vae.init(rng=rng_key, batch=[dummy_x,dummy_y])
best_params = pickle.load(open(os.path.join("output/weights", "best_ckpt.pkl"), "rb"))


def infer(num, state=state):
    x = jnp.array(np.random.rand(1, 1, 28, 28))
    x = jnp.array(x)
    y = jnp.array([num])
    y_1 = jnp.tile(y[:, None, None, None], (1, 1, 28, 28))
    x = jnp.concatenate((x, y_1), axis=1)
    x = jnp.transpose(x, (0, 2, 3, 1))
    res, state = vae.apply(best_params, state, rng_key, [x, y])
    
    return np.array(res[0][0].reshape(28,28))