import math
from dataclasses import dataclass
from typing import Self, final

import jax
import polars as pl


class BaseLoader:
    __prng_key: jax.Array | None = None

    @final
    def init(self, prng_key: jax.Array) -> Self:
        """Replaces the current PRNG key with a new PRNG key for each epoch

        Args:
            prng_key (jax.Array): new PRNG key

        Note:
            You are not allowed to override this function in your child classes!
        """
        self.__prng_key = prng_key.copy()
        return self

    @final
    @property
    def prng_key(self) -> jax.Array:
        """Returns the current PRNG key

        Note:
            You are not allowed to override this function in your child classes!
        """

        if self.__prng_key is None:
            raise Exception

        return self.__prng_key

    def __iter__(self) -> Self:
        """Prepares for batch iteration"""

        raise NotImplementedError

    def __len__(self) -> int:
        """Returns the number of batches

        Returns:
            int: The number of batches
        """

        raise NotImplementedError

    def __next__(self) -> tuple[jax.Array, jax.Array]:
        """Returns data from the current batch

        Returns:
            jax.Array: The input data.
            jax.Array: The target data.
        """

        raise NotImplementedError


@dataclass
class MiniBatchLoader(BaseLoader):
    df_DATA: pl.DataFrame
    batch_size: int

    def __iter__(self) -> Self:
        """Prepares for batch iteration"""

        # Num. of data
        self.data_size = self.df_DATA.height

        # Num. of batch
        self.batch_num = math.ceil(self.data_size / self.batch_size)

        # Shuffle rows of data
        self.shuffled_indices = jax.random.permutation(self.prng_key, self.data_size)
        self.X, self.y = (
            jax.device_put(self.df_DATA[:, :-1].to_numpy())[self.shuffled_indices],
            jax.device_put(self.df_DATA[:, -1:].to_numpy())[self.shuffled_indices],
        )

        # Initialize batch index
        self.batch_index = 0

        return self

    def __len__(self) -> int:
        """Returns the number of batches

        Returns:
            int: The number of batches
        """

        return self.batch_num

    def __next__(self) -> tuple[jax.Array, jax.Array]:
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

            return X, y
