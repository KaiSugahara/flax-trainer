from flax import linen as nn


class RegressionMLP(nn.Module):
    """Multilayer perceptron model for regression task

    Attributes:
        layer_size (list[int]): Number of nodes per layer.
    """

    layer_sizes: list[int]

    def setup(self):
        self.denses = [nn.Dense(features=node_num) for node_num in self.layer_sizes]

    def __call__(self, X):
        for fn in self.denses[:-1]:
            X = fn(X)
            X = nn.relu(X)
        X = self.denses[-1](X)
        return X
