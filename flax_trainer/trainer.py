from dataclasses import dataclass
from typing import Callable, Self, Tuple

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from tqdm import tqdm

from .evaluator import BaseEvaluator
from .loader import BaseLoader
from .logger import Logger


@dataclass
class Trainer:
    """Trainer for Flax model

    Attributes:
        model (nn.Module): The Flax model you are training.
        optimizer (optax.GradientTransformation): The optax optimizer.
        train_loader (BaseLoader): The data loader used in training
        loss_fn (Callable[[nn.Module], Callable[[nn.FrozenDict, jax.Array, jax.Array], jax.Array]]): Loss function evaluated in training
        test_evaluator (BaseEvaluator): (Optional) The evaluator for testing. Defaults to None.
        early_stopping_patience (int): (Optional) Number of epochs with no improvement after which training will be stopped. Defaults to 0.
        epoch_num (int): (Optional) Number of training iterations. Defaults to 16.
        seed (int): (Optional) Random seed used for initialization or sampling. Defaults to 0.
    """

    model: nn.Module
    optimizer: optax.GradientTransformation
    train_loader: BaseLoader
    loss_fn: Callable[[nn.Module], Callable[[nn.FrozenDict, jax.Array, jax.Array], jax.Array]]
    test_evaluator: BaseEvaluator | None = None
    early_stopping_patience: int = 0
    epoch_num: int = 16
    seed: int = 0

    def __get_prng_key(self) -> jax.Array:
        """Generates and returns new PRNG key from current PRNG key

        Returns:
            jax.Array: new PRNG key
        """

        self._prng_key, return_key = jax.random.split(getattr(self, "_prng_key", jax.random.PRNGKey(self.seed)))
        return return_key

    def __init_model_params(self) -> nn.FrozenDict:
        """Initializes and returns model parameters

        Returns:
            nn.FrozenDict: model parameters
        """

        X, _ = next(iter(self.train_loader.init(self.__get_prng_key())))
        return self.model.init(self.__get_prng_key(), X)["params"]

    def fit(self, model_params: nn.FrozenDict | None = None) -> Self:
        """Trains the model on training data.

        Args:
            model_params (nn.FrozenDict): (Optional) You can set your model parameters as initial value. Defaults to None.
        """

        @jax.jit
        def step_batch(
            opt_state: optax.OptState,
            model_params: nn.FrozenDict,
            X: jax.Array,
            y: jax.Array,
        ) -> Tuple[optax.OptState, nn.FrozenDict, jax.Array]:
            """Updates model parameters on the training batch

            Args:
                opt_state (optax.OptState): The current state of optimizer.
                model_params (nn.FrozenDict): The current model parameters.
                X (jax.Array): The training input data.
                y (jax.Array): The target data.

            Returns:
                optax.OptState: The updated state of the optimizer.
                nn.FrozenDict: The updated model parameters.
                jax.Array: The value of the loss on the batch.
            """

            batch_loss_value, grads = jax.value_and_grad(self.loss_fn(self.model))(model_params, X, y)
            updates, opt_state = self.optimizer.update(grads, opt_state, model_params)
            model_params = optax.apply_updates(model_params, updates)  # type: ignore
            return opt_state, model_params, batch_loss_value

        def step_epoch(
            opt_state: optax.OptState, model_params: nn.FrozenDict, epoch_i: int
        ) -> Tuple[optax.OptState, nn.FrozenDict, float]:
            """Updates model parameters on the epoch

            Args:
                opt_state (optax.OptState): The current state of optimizer.
                model_params (nn.FrozenDict): The current model parameters.
                epoch_i (int): The current epoch index.

            Returns:
                optax.OptState: The updated state of the optimizer.
                nn.FrozenDict: The updated model parameters.
                float: The value of the loss on the epoch.
            """

            batch_loss_value_buff: list[jax.Array] = []

            # Train for every batch defined by loader
            pbar = tqdm(self.train_loader.init(self.__get_prng_key()))
            pbar.set_description(f"[TRAIN {str(epoch_i).zfill(3)}]")
            for X, y in pbar:
                opt_state, model_params, batch_loss_value = step_batch(opt_state, model_params, X, y)
                batch_loss_value_buff.append(batch_loss_value)
                pbar.set_postfix({"batch_loss": batch_loss_value})

            # Calculate the average of batch loss values
            epoch_loss_value = float(jnp.mean(jnp.array(batch_loss_value_buff)))

            return opt_state, model_params, epoch_loss_value

        def test(model_params: nn.FrozenDict, epoch_i: int) -> None:
            """Calculates and logs test loss/scores

            Args:
                model_params (nn.FrozenDict): The current model parameters.
                epoch_i (int): The current epoch index.
            """
            if self.test_evaluator is not None:
                test_loss, test_metrics = self.test_evaluator.evaluate(self.model, model_params)
                print(f"[TEST  {str(epoch_i).zfill(3)}]", f"loss={test_loss}")
                self.logger.log_test_loss(test_loss, epoch_i)
                self.logger.log_test_metrics(test_metrics, epoch_i)

        # Initialize logger
        self.logger = Logger()

        # Initialize model params. and optimizer state
        if model_params is None:
            model_params = self.__init_model_params()
        opt_state = self.optimizer.init(model_params)

        # Test
        test(model_params, epoch_i=0)

        # Iterate training {epoch_num} times
        for epoch_i in range(1, self.epoch_num + 1):
            # Train
            opt_state, model_params, epoch_loss_value = step_epoch(opt_state, model_params, epoch_i)
            self.logger.log_train_loss(epoch_loss_value, epoch_i)

            # Test
            test(model_params, epoch_i)

            # Check early stopping
            if (
                self.test_evaluator
                and self.early_stopping_patience > 0
                and self.logger.early_stopping(self.early_stopping_patience, epoch_i)
            ):
                break

        # Save model-params as instance variable
        self.model_params = model_params.copy()

        return self
