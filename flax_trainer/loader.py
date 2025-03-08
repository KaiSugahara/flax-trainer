import math
from typing import Self

import jax
import polars as pl
from flax import nnx


class BaseLoader:
    def setup_epoch(self) -> Self:
        """Data preparation before the start of every epoch"""

        raise NotImplementedError

    def __iter__(self) -> Self:
        """Initialize iteration"""

        raise NotImplementedError

    def __len__(self) -> int:
        """Returns the number of batches

        Returns:
            int: The number of batches
        """

        raise NotImplementedError

    def __next__(self) -> tuple[tuple[jax.Array, ...], jax.Array]:
        """Returns data from the current batch

        Returns:
            jax.Array: The input data.
            jax.Array: The target data.
        """

        raise NotImplementedError


class MiniBatchLoader(BaseLoader):
    def __init__(
        self,
        dataset_df: pl.DataFrame,
        batch_size: int,
        rngs: nnx.Rngs,
    ):
        self.dataset_df = dataset_df
        self.batch_size = batch_size
        self.rngs = rngs

    def setup_epoch(self) -> Self:
        """Data preparation before the start of every epoch"""

        # Num. of data
        self.data_size = self.dataset_df.height

        # Num. of batch
        self.batch_num = math.ceil(self.data_size / self.batch_size)

        # Shuffle rows of data
        self.shuffled_indices = jax.random.permutation(self.rngs(), self.data_size)
        self.X, self.y = (
            jax.device_put(self.dataset_df[:, :-1].to_numpy())[self.shuffled_indices],
            jax.device_put(self.dataset_df[:, -1:].to_numpy())[self.shuffled_indices],
        )

        return self

    def __iter__(self) -> Self:
        """Initialize iteration"""

        # Initialize batch index
        self.batch_index = 0

        return self

    def __len__(self) -> int:
        """Returns the number of batches

        Returns:
            int: The number of batches
        """

        return self.batch_num

    def __next__(self) -> tuple[tuple[jax.Array, ...], jax.Array]:
        """Returns data from the current batch

        Returns:
            jax.Array: The input data.
            jax.Array: The target data.
        """

        if self.batch_index >= self.batch_num:
            raise StopIteration()

        else:
            # Extract the {batch_index}-th mini-batch
            start_index = self.batch_size * self.batch_index
            slice_size = min(self.batch_size, (self.data_size - start_index))
            X, y = (
                jax.lax.dynamic_slice_in_dim(self.X, start_index, slice_size),
                jax.lax.dynamic_slice_in_dim(self.y, start_index, slice_size),
            )

            # Update batch index
            self.batch_index += 1

            return (X,), y
