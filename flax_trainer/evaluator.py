import jax
import jax.numpy as jnp
import polars as pl
from flax import linen as nn
from flax import nnx


class BaseEvaluator:
    def evaluate(self, model: nnx.Module) -> tuple[float, dict[str, float]]:
        """Calculates test loss and metrics.

        Args:
            model (nn.Module): The Flax model you are training.

        Returns:
            float: The test loss.
            dict[str, float]: The test metrics.
        """

        raise NotImplementedError


class RegressionEvaluator(BaseEvaluator):
    """Evaluator for regression task

    Attributes:
        df_DATA (pl.DataFrame): The testing data.
    """

    def __init__(self, df_DATA: pl.DataFrame):
        self.X = jax.device_put(df_DATA[:, :-1].to_numpy())
        self.y = jax.device_put(df_DATA[:, -1:].to_numpy())

    def evaluate(self, model: nn.Module) -> tuple[float, dict[str, float]]:
        """Calculates test loss and metrics.

        Args:
            model (nn.Module): The Flax model you are training.

        Returns:
            float: The test loss.
            dict[str, float]: The test metrics.
        """

        def calc_mse(X: jax.Array, y: jax.Array) -> jax.Array:
            # Prediction
            pred = model(X)

            # MSE
            loss = jnp.mean((pred - y) ** 2)

            return loss

        # MSE
        mse = float(calc_mse(self.X, self.y))

        return mse, {"mse": mse}
