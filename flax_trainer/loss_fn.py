import jax
import jax.numpy as jnp
from flax import nnx


def mean_squared_error(model: nnx.Module, X: jax.Array, y: jax.Array) -> jax.Array:
    # Prediction
    pred = model(X)  # type: ignore

    # MSE
    loss = jnp.mean((pred - y) ** 2)

    return loss
