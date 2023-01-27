import sys
sys.path.append(".")
import jax
import haiku as hk
import optax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
import pickle
import os

from backend.src.model.cvae import vae
from backend.src.datasets.mnist import train_loader, val_loader


def vae_loss(params: hk.Params, state, batch):
    loss_input, state = vae.apply(params, state, rng_key, batch)
    recons = loss_input[0]
    ori_image = loss_input[1]
    mean = loss_input[2]
    var = loss_input[3]
    recons_loss = optax.l2_loss(recons, ori_image[:,:,:,:1])
    kld_loss = jnp.mean(-0.5 * jnp.sum(1 + var - mean**2 -
                        jnp.exp(var), axis=1), axis=0)
    loss = recons_loss + kld_loss* 0.00025
    return loss.mean()

vae = hk.transform_with_state(vae)
dummy_x = jnp.array(np.random.rand(8, 28, 28, 2))
dummy_y = jnp.array(np.random.rand(8))
rng_key = jax.random.PRNGKey(42)
params, state = vae.init(rng=rng_key, batch=[dummy_x,dummy_y])

optimizer = optax.adamw(learning_rate=0.001)
opt_state = optimizer.init(params)

@jax.jit
def step(params, opt_state, state, batch):
    loss_value, grads = jax.value_and_grad(vae_loss)(params, state, batch)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss_value

for epoch in range(1):
    for x, y in tqdm(train_loader):
        x = jnp.array(x)
        y = jnp.array(y)
        y_1 = jnp.tile(y[:,None,None,None], (1, 1, 28, 28))
        x = jnp.concatenate((x,y_1), axis=1)
        x = jnp.transpose(x, (0, 2, 3, 1))
        params, opt_state, loss_value = step(params, opt_state, state,[x ,y])
    
    pickle.dump(params, open(os.path.join("backend/output/weights", "best_ckpt.pkl"), "wb"))
    [x, y] = next(iter(val_loader))
    x = jnp.array(x)
    y = jnp.array([2])
    y_1 = jnp.tile(y[:, None, None, None], (1, 1, 28, 28))
    x = jnp.concatenate((x, y_1), axis=1)
    x = jnp.transpose(x, (0, 2, 3, 1))

    best_params = pickle.load(open(os.path.join("backend/output/weights", "best_ckpt.pkl"), "rb"))
    res, state = vae.apply(best_params, state, rng_key, [x, y])
    plt.imshow(res[0][0].reshape(28,28))
    plt.show()
