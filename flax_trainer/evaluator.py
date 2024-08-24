from dataclasses import dataclass

import jax
import jax.numpy as jnp
import polars as pl
from flax import linen as nn


class BaseEvaluator:
    def evaluate(self, model: nn.Module, model_params: nn.FrozenDict) -> tuple[float, dict[str, float]]:
        """Calculates test loss and metrics.

        Args:
            model (nn.Module): The Flax model you are training.
            model_params (nn.FrozenDict): The current model parameters.

        Returns:
            float: The test loss.
            dict[str, float]: The test metrics.
        """

        raise NotImplementedError


@dataclass
class RegressionEvaluator(BaseEvaluator):
    """Evaluator for regression task

    Attributes:
        df_DATA (pl.DataFrame): The testing data.
    """

    df_DATA: pl.DataFrame

    def __post_init__(self):
        self.X = jax.device_put(self.df_DATA[:, :-1].to_numpy())
        self.y = jax.device_put(self.df_DATA[:, -1:].to_numpy())

    def evaluate(self, model: nn.Module, model_params: nn.FrozenDict) -> tuple[float, dict[str, float]]:
        """Calculates test loss and metrics.

        Args:
            model (nn.Module): The Flax model you are training.
            model_params (nn.FrozenDict): The current model parameters.

        Returns:
            float: The test loss.
            dict[str, float]: The test metrics.
        """

        def calc_mse(model_params: nn.FrozenDict, X: jax.Array, y: jax.Array) -> jax.Array:
            # Prediction
            pred = model.apply({"params": model_params}, X)

            # MSE
            loss = jnp.mean((pred - y) ** 2)

            return loss

        # MSE
        mse = float(calc_mse(model_params, self.X, self.y))

        return mse, {"mse": mse}
