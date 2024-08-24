from typing import Callable

import jax
import jax.numpy as jnp
from flax import linen as nn


def mean_squared_error(
    model: nn.Module,
) -> Callable[[nn.FrozenDict, jax.Array, jax.Array], jax.Array]:
    @jax.jit
    def inner(params: nn.FrozenDict, X: jax.Array, y: jax.Array):
        # Prediction
        pred = model.apply({"params": params}, X)

        # MSE
        loss = jnp.mean((pred - y) ** 2)

        return loss

    return inner
