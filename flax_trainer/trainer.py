from dataclasses import dataclass
from typing import Callable, Self

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from tqdm import tqdm

from .evaluator import BaseEvaluator
from .loader import BaseLoader
from .logger import Logger


@dataclass
class Trainer:
    """Trainer for Flax model

    Attributes:
        model (nnx.Module): The Flax model you are training.
        optimizer (optax.GradientTransformation): The optax optimizer.
        train_loader (BaseLoader): The data loader used in training
        loss_fn (Callable[[nnx.Module], Callable[[nn.FrozenDict, jax.Array, jax.Array], jax.Array]]): Loss function evaluated in training
        test_evaluator (BaseEvaluator): (Optional) The evaluator for testing. Defaults to None.
        early_stopping_patience (int): (Optional) Number of epochs with no improvement after which training will be stopped. Defaults to 0.
        epoch_num (int): (Optional) Number of training iterations. Defaults to 16.
    """

    model: nnx.Module
    optimizer: optax.GradientTransformation
    train_loader: BaseLoader
    loss_fn: Callable[[nnx.Module, jax.Array, jax.Array], jax.Array]
    test_evaluator: BaseEvaluator | None = None
    early_stopping_patience: int = 0
    epoch_num: int = 16

    def fit(self) -> Self:
        """Trains the model on training data"""

        @nnx.jit
        def step_batch(
            opt_state: nnx.Optimizer,
            X: jax.Array,
            y: jax.Array,
        ) -> jax.Array:
            """Updates model parameters on the training batch

            Args:
                opt_state (nnx.OptState): The current state of optimizer.
                X (jax.Array): The training input data.
                y (jax.Array): The target data.

            Returns:
                jax.Array: The value of the loss on the batch.
            """

            batch_loss, grads = nnx.value_and_grad(self.loss_fn)(self.model, X, y)
            opt_state.update(grads)

            return batch_loss

        def step_epoch(opt_state: nnx.Optimizer, epoch_i: int) -> float:
            """Updates model parameters on the epoch

            Args:
                opt_state (nnx.OptState): The current state of optimizer.
                epoch_i (int): The current epoch index.

            Returns:
                optax.OptState: The updated state of the optimizer.
                nn.FrozenDict: The updated model parameters.
                float: The value of the loss on the epoch.
            """

            batch_loss_buff: list[jax.Array] = []

            # Train for every batch defined by loader
            pbar = tqdm(iter(self.train_loader))
            pbar.set_description(f"[TRAIN {str(epoch_i).zfill(3)}]")
            for X, y in pbar:
                batch_loss = step_batch(opt_state, X, y)
                batch_loss_buff.append(batch_loss)
                pbar.set_postfix({"batch_loss": batch_loss})

            # Calculate the average of batch loss values
            epoch_loss = float(jnp.mean(jnp.array(batch_loss_buff)))

            return epoch_loss

        def test(epoch_i: int) -> None:
            """Calculates and logs test loss/scores

            Args:
                epoch_i (int): The current epoch index.
            """
            if self.test_evaluator is not None:
                test_loss, test_metrics = self.test_evaluator.evaluate(self.model)
                print(f"[TEST  {str(epoch_i).zfill(3)}]", f"loss={test_loss}")
                self.logger.log_test_loss(test_loss, epoch_i)
                self.logger.log_test_metrics(test_metrics, epoch_i)

        # Initialize logger
        self.logger = Logger()

        # Initialize optimizer
        opt_state = nnx.Optimizer(model=self.model, tx=self.optimizer)

        # Test
        test(epoch_i=0)

        # Iterate training {epoch_num} times
        for epoch_i in range(1, self.epoch_num + 1):
            # Train
            epoch_loss = step_epoch(opt_state, epoch_i)
            self.logger.log_train_loss(epoch_loss, epoch_i)

            # Test
            test(epoch_i)

            # Check early stopping
            if (
                (self.test_evaluator is not None)
                and self.early_stopping_patience > 0
                and self.logger.early_stopping(self.early_stopping_patience, epoch_i)
            ):
                break

        return self
