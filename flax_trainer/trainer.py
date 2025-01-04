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
        loss_fn (Callable[[nnx.Module, jax.Array, jax.Array], jax.Array]): Loss function evaluated in training
        test_evaluator (BaseEvaluator): (Optional) The evaluator for testing. Defaults to None.
        early_stopping_patience (int): (Optional) Number of epochs with no improvement after which training will be stopped. Defaults to 0.
        epoch_num (int): (Optional) Number of training iterations. Defaults to 16.
    """

    model: nnx.Module
    optimizer: optax.GradientTransformation
    train_loader: BaseLoader
    loss_fn: Callable[[nnx.Module, tuple[jax.Array, ...], jax.Array], jax.Array]
    test_evaluator: BaseEvaluator | None = None
    early_stopping_patience: int = 0
    epoch_num: int = 16

    def fit(self) -> Self:
        """Trains the model on training data"""

        @nnx.jit
        def step_batch(
            model: nnx.Module,
            opt_state: nnx.Optimizer,
            Xs: tuple[jax.Array, ...],
            y: jax.Array,
        ) -> jax.Array:
            """Updates model parameters on the training batch

            Args:
                model (nnx.Module): The Flax model you are training.
                opt_state (nnx.OptState): The current state of optimizer.
                Xs (tuple[jax.Array, ...]): The training input data(s).
                y (jax.Array): The target data.

            Returns:
                jax.Array: The value of the loss on the batch.
            """

            batch_loss, grads = nnx.value_and_grad(self.loss_fn)(model, Xs, y)
            opt_state.update(grads)

            return batch_loss

        def step_epoch(epoch_i: int) -> float:
            """Updates model parameters on the epoch

            Args:
                epoch_i (int): The current epoch index.

            Returns:
                float: The value of the loss on the epoch.
            """

            batch_loss_buff: list[jax.Array] = []

            # Train for every batch defined by loader
            pbar = tqdm(iter(self.train_loader))
            pbar.set_description(f"[TRAIN {str(epoch_i).zfill(3)}]")
            for Xs, y in pbar:
                batch_loss = step_batch(self.model, self.opt_state, Xs, y)
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
            if self.test_evaluator is None:
                return None

            # Calculate and log test loss/scores
            test_loss, test_metrics = self.test_evaluator.evaluate(self.model)
            print(f"[TEST  {str(epoch_i).zfill(3)}]", f"loss={test_loss}")
            self.logger.log_test_loss(test_loss, epoch_i)
            self.logger.log_test_metrics(test_metrics, epoch_i)

            # Update best parameters if test loss is best
            if epoch_i == self.logger.best_epoch_i:
                _, self.__best_state = nnx.split(self.model)

        # Initialize logger
        self.logger = Logger()

        # Initialize optimizer
        self.opt_state = nnx.Optimizer(model=self.model, tx=self.optimizer)

        # Initialize best state
        self.__best_state: nnx.graph.GraphState | None = None

        # Test
        test(epoch_i=0)

        # Iterate training {epoch_num} times
        for epoch_i in range(1, self.epoch_num + 1):
            # Train
            epoch_loss = step_epoch(epoch_i)
            self.logger.log_train_loss(epoch_loss, epoch_i)

            # Test
            test(epoch_i)

            # Check early stopping
            if (
                (self.test_evaluator is not None)
                and (self.early_stopping_patience > 0)
                and (epoch_i - self.logger.best_epoch_i) >= self.early_stopping_patience
            ):
                break

        return self

    @property
    def best_model(self) -> nnx.Module | None:
        best_state = getattr(self, "_Trainer__best_state", None)
        if best_state is None:
            return None
        graphdef, _ = nnx.split(self.model)
        return nnx.merge(graphdef, best_state)

    @property
    def best_state_dict(self) -> nnx.Module | None:
        best_state = getattr(self, "_Trainer__best_state", None)
        if best_state is None:
            return None
        return best_state.to_pure_dict()
