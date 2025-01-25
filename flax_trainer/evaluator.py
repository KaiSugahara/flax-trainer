import jax
import jax.numpy as jnp
import polars as pl
from flax import linen as nn
from flax import nnx


class BaseEvaluator:
    def evaluate(self, model: nnx.Module) -> tuple[float, dict[str, float]]:
        """Calculate loss and metrics for the model

        Args:
            model (nn.Module): The Flax model you are training.

        Returns:
            float: The loss.
            dict[str, float]: The metrics.
        """

        raise NotImplementedError


class RegressionEvaluator(BaseEvaluator):
    """Evaluator for regression task"""

    def __init__(self, dataset_df: pl.DataFrame):
        """Initialize instance

        Args:
            dataset_df (pl.DataFrame): The testing data.
        """

        self.Xs = (jax.device_put(dataset_df[:, :-1].to_numpy()),)
        self.y = jax.device_put(dataset_df[:, -1:].to_numpy())

    def evaluate(self, model: nn.Module) -> tuple[float, dict[str, float]]:
        """Calculate loss and metrics for the model

        Args:
            model (nn.Module): The Flax model you are training.

        Returns:
            float: The loss.
            dict[str, float]: The metrics.
        """

        def calc_mse(Xs: tuple[jax.Array, ...], y: jax.Array) -> jax.Array:
            # Prediction
            pred = model(*Xs)

            # MSE
            loss = jnp.mean((pred - y) ** 2)

            return loss

        # MSE
        mse = float(calc_mse(self.Xs, self.y))

        return mse, {"mse": mse}
