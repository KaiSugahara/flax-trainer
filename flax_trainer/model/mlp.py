import jax
from flax import nnx


class RegressionMLP(nnx.Module):
    """Multilayer perceptron model for regression task"""

    def __init__(self, input_layer_dim: int, hidden_layer_dims: list[int], output_layer_dim: int, rngs: nnx.Rngs):
        """Initialize model layer and parameters

        Args:
            input_layer_dim (int): the number of neurons in the input layer.
            hidden_layer_dims (list[int]): the numbers of neurons in the hidden layers.
            output_layer_dim (int): the number of neurons in the output layer.
            rngs (nnx.Rngs): rng key.
        """

        dims = [input_layer_dim] + hidden_layer_dims + [output_layer_dim]
        self.linears = [
            nnx.Linear(in_features=dims[i], out_features=dims[i + 1], rngs=rngs) for i in range(len(dims) - 1)
        ]

    def __call__(self, *Xs: jax.Array) -> jax.Array:
        X = Xs[0]
        for linear in self.linears[:-1]:
            X = linear(X)
            X = nnx.relu(X)
        X = self.linears[-1](X)
        return X
