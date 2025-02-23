import os
import pickle
import tempfile
from dataclasses import dataclass

import mlflow
import numpy as np
from mlflow import ActiveRun
from mlflow.config import enable_async_logging

enable_async_logging()


@dataclass
class Logger:
    """Logger

    Attributes:
        active_run (ActiveRun): MLFlow's run state
    """

    active_run: ActiveRun | None

    def log_train_loss(self, value: float, epoch_i: int):
        """Logs the training loss for the epoch.

        Args:
            value (float): The value of the training loss
            epoch_i (int): The current epoch index.
        """

        if isinstance(self.active_run, ActiveRun):
            mlflow.log_metric("train_loss", value, step=epoch_i)

    def log_valid_loss(self, value: float, epoch_i: int):
        """Logs the valid loss for the epoch.

        Args:
            value (float): The value of the valid loss
            epoch_i (int): The current epoch index.
        """

        if isinstance(self.active_run, ActiveRun):
            mlflow.log_metric("valid_loss", value, step=epoch_i)

        # Update best epoch
        if self.best_valid_loss >= value:
            self._best_epoch_i = epoch_i
            self._best_valid_loss = value
            mlflow.log_metric("best_valid_loss", value, step=epoch_i)

    def log_valid_metrics(self, metrics: dict[str, float], epoch_i: int):
        """Logs the valid metrics for the epoch.

        Args:
            metrics (dict[str, float]): The valid scores by metrics.
            epoch_i (int): The current epoch index.
        """

        if isinstance(self.active_run, ActiveRun):
            mlflow.log_metrics({f"valid_{key}": value for key, value in metrics.items()}, step=epoch_i)

    def log_best_state_dict(self, best_state_dict: dict) -> None:
        """Logs the best model state dict

        Args:
            best_state_dict (dict): The best state dict
        """

        if isinstance(self.active_run, ActiveRun):
            with tempfile.TemporaryDirectory() as temp_dir_path:
                temp_file_path = os.path.join(temp_dir_path, "best_state_dict.pickle")
                with open(temp_file_path, "wb") as f:
                    pickle.dump(best_state_dict, f)
                mlflow.log_artifact(temp_file_path)

    @property
    def best_epoch_i(self) -> int:
        return getattr(self, "_best_epoch_i", 0)

    @property
    def best_valid_loss(self) -> float:
        return getattr(self, "_best_valid_loss", np.inf)
