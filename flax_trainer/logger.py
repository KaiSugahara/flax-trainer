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

        # Update best epoch
        if self.test_metrics[self.best_epoch_i]["loss"] >= value:
            self._best_epoch_i = epoch_i

    def log_test_metrics(self, metrics: dict[str, float], epoch_i: int):
        """Logs the testing metrics for the epoch.

        Args:
            metrics (dict[str, float]): The testing scores by metrics.
            epoch_i (int): The current epoch index.
        """

        for key, value in metrics.items():
            self.test_metrics.setdefault(epoch_i, {})[key] = value

    @property
    def best_epoch_i(self) -> int:
        return getattr(self, "_best_epoch_i", 0)
