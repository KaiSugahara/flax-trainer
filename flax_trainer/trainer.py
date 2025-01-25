from dataclasses import dataclass
from typing import Callable, Generic, Self, TypeVar

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from mlflow import ActiveRun
from tqdm import tqdm

from .evaluator import BaseEvaluator
from .loader import BaseLoader
from .logger import Logger

Model = TypeVar("Model", bound=nnx.Module)


class NotFittedError(Exception): ...


@dataclass
class Trainer(Generic[Model]):
    """Trainer for Flax model

    Attributes:
        model (Model): The Flax model you are training.
        optimizer (optax.GradientTransformation): The optax optimizer.
        train_loader (BaseLoader): The data loader used in training
        loss_fn (Callable[[Model, jax.Array, jax.Array], jax.Array]): Loss function evaluated in training
        valid_evaluator (BaseEvaluator): (Optional) The evaluator for validation. Defaults to None.
        early_stopping_patience (int): (Optional) Number of epochs with no improvement after which training will be stopped. Defaults to 0.
        epoch_num (int): (Optional) Number of training iterations. Defaults to 16.
        active_run (ActiveRun): (Optional) MLFlow's run state.
    """

    model: Model
    optimizer: optax.GradientTransformation
    train_loader: BaseLoader
    loss_fn: Callable[[Model, tuple[jax.Array, ...], jax.Array], jax.Array]
    valid_evaluator: BaseEvaluator | None = None
    early_stopping_patience: int = 0
    epoch_num: int = 16
    active_run: ActiveRun | None = None

    def fit(self) -> Self:
        """Trains the model on training data"""

        @nnx.jit
        def step_batch(
            model: Model,
            opt_state: nnx.Optimizer,
            Xs: tuple[jax.Array, ...],
            y: jax.Array,
        ) -> jax.Array:
            """Updates model parameters on the training batch

            Args:
                model (Model): The Flax model you are training.
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

        def valid_and_check_early_stopping(epoch_i: int) -> bool:
            """Logs valid loss/scores and check early stopping

            Args:
                epoch_i (int): The current epoch index.

            Returns:
                bool: Flag indicating whether or not early stop
            """
            if self.valid_evaluator is None:
                return False

            # Calculate and log valid loss/scores
            loss, metrics = self.valid_evaluator.evaluate(self.model)
            print(f"[VALID {str(epoch_i).zfill(3)}]:", f"{loss=}, {metrics=}")
            self.logger.log_valid_loss(loss, epoch_i)
            self.logger.log_valid_metrics(metrics, epoch_i)

            # Update best parameters if valid loss is best
            if epoch_i == self.logger.best_epoch_i:
                _, self.__best_state = nnx.split(self.model)

            # Check early stopping
            early_stopping_flag = (self.early_stopping_patience > 0) and (
                (epoch_i - self.logger.best_epoch_i) >= self.early_stopping_patience
            )

            return early_stopping_flag

        # Initialize logger
        self.logger = Logger(active_run=self.active_run)

        # Initialize optimizer
        self.opt_state = nnx.Optimizer(model=self.model, tx=self.optimizer)

        # Initialize best state
        self.__best_state: nnx.graph.GraphState | None = None

        # Validation
        valid_and_check_early_stopping(epoch_i=0)

        # Iterate training {epoch_num} times
        for epoch_i in range(1, self.epoch_num + 1):
            # Train
            epoch_loss = step_epoch(epoch_i)
            self.logger.log_train_loss(epoch_loss, epoch_i)

            # Validation and Check early stopping
            if valid_and_check_early_stopping(epoch_i):
                break

        return self

    @property
    def best_model(self) -> Model:
        best_state = getattr(self, "_Trainer__best_state", None)
        if best_state is None:
            raise NotFittedError()
        graphdef, _ = nnx.split(self.model)
        return nnx.merge(graphdef, best_state)

    @property
    def best_state_dict(self) -> dict:
        best_state = getattr(self, "_Trainer__best_state", None)
        if best_state is None:
            raise NotFittedError()
        return best_state.to_pure_dict()
