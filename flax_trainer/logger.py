from dataclasses import dataclass, field


@dataclass
class Logger:
    """Logger

    Attributes:
        train_metrics (dict[int, dict[str, float]]): Metric scores of the model in the training set by epoch
        test_metrics (dict[int, dict[str, float]]): Metric scores of the model in the testing set by epoch
    """

    train_metrics: dict[int, dict[str, float]] = field(default_factory=dict)
    test_metrics: dict[int, dict[str, float]] = field(default_factory=dict)

    def log_train_loss(self, value: float, epoch_i: int):
        """Logs the training loss for the epoch.

        Args:
            value (float): The value of the training loss
            epoch_i (int): The current epoch index.
        """

        self.train_metrics.setdefault(epoch_i, {})["loss"] = value

    def log_test_loss(self, value: float, epoch_i: int):
        """Logs the testing loss for the epoch.

        Args:
            value (float): The value of the testing loss
            epoch_i (int): The current epoch index.
        """

        self.test_metrics.setdefault(epoch_i, {})["loss"] = value

    def log_test_metrics(self, metrics: dict[str, float], epoch_i: int):
        """Logs the testing metrics for the epoch.

        Args:
            metrics (dict[str, float]): The testing scores by metrics.
            epoch_i (int): The current epoch index.
        """

        for key, value in metrics.items():
            self.test_metrics.setdefault(epoch_i, {})[key] = value

    def early_stopping(self, early_stopping_patience: int, epoch_i: int) -> bool:
        """Checks if the conditions for EARLY STOPPING are met.

        Args:
            early_stopping_patience (int): Number of epochs with no improvement after which training will be stopped.
            epoch_i (int): The current epoch index.

        Returns:
            _type_: Whether the conditions for EARLY STOPPING have been met
        """

        best_epoch_i: int = getattr(self, "_best_epoch_i", 0)
        best_test_loss: float = getattr(self, "_best_test_loss", self.test_metrics[0]["loss"])
        current_test_loss = self.test_metrics[epoch_i]["loss"]

        if best_test_loss >= current_test_loss:
            self._best_epoch_i = epoch_i
            self._best_test_loss = current_test_loss

        else:
            return (epoch_i - best_epoch_i) >= early_stopping_patience

        return False
