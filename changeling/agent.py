"""S1 substrate: plain GRU, pure JAX. Params are a flat dict pytree."""
import jax
import jax.numpy as jnp

from . import IN_DIM, N_ACT


def init_gru(key, hidden=128, in_dim=IN_DIM, out_dim=N_ACT):
    ks = jax.random.split(key, 7)
    s_in, s_h = 1.0 / jnp.sqrt(in_dim), 1.0 / jnp.sqrt(hidden)

    def mat(k, shape, scale):
        return jax.random.normal(k, shape) * scale

    return {
        "Wxz": mat(ks[0], (in_dim, hidden), s_in), "Whz": mat(ks[1], (hidden, hidden), s_h),
        "bz": jnp.zeros(hidden),
        "Wxr": mat(ks[2], (in_dim, hidden), s_in), "Whr": mat(ks[3], (hidden, hidden), s_h),
        "br": jnp.zeros(hidden),
        "Wxh": mat(ks[4], (in_dim, hidden), s_in), "Whh": mat(ks[5], (hidden, hidden), s_h),
        "bh": jnp.zeros(hidden),
        "Wo": mat(ks[6], (hidden, out_dim), s_h), "bo": jnp.zeros(out_dim),
    }


def gru_step(p, h, x):
    z = jax.nn.sigmoid(x @ p["Wxz"] + h @ p["Whz"] + p["bz"])
    r = jax.nn.sigmoid(x @ p["Wxr"] + h @ p["Whr"] + p["br"])
    hh = jnp.tanh(x @ p["Wxh"] + (r * h) @ p["Whh"] + p["bh"])
    h = (1.0 - z) * h + z * hh
    return h, h @ p["Wo"] + p["bo"]


def hidden_size(params):
    return params["bz"].shape[0]
