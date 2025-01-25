from dataclasses import dataclass

import mlflow
import numpy as np
from mlflow import ActiveRun


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

    def log_valid_metrics(self, metrics: dict[str, float], epoch_i: int):
        """Logs the valid metrics for the epoch.

        Args:
            metrics (dict[str, float]): The valid scores by metrics.
            epoch_i (int): The current epoch index.
        """

        for key, value in metrics.items():
            mlflow.log_metric(f"valid_{key}", value, step=epoch_i)

    @property
    def best_epoch_i(self) -> int:
        return getattr(self, "_best_epoch_i", 0)

    @property
    def best_valid_loss(self) -> float:
        return getattr(self, "_best_valid_loss", np.inf)
