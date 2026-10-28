"""Early stopping monitor with serializable state for training resumption."""

from typing import Any, Dict, Optional


class EarlyStopping:
    """
    Early stopping handler that monitors a validation metric and halts training
    when no improvement is observed after a given patience window.
    """

    def __init__(
        self,
        monitor: str = "val_accuracy",
        patience: int = 7,
        min_delta: float = 0.0001,
        mode: str = "max"
    ):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode.lower()

        if self.mode not in ("min", "max"):
            raise ValueError(f"Mode must be 'min' or 'max', got {mode}")

        self.best_score: Optional[float] = None
        self.counter: int = 0
        self.early_stop: bool = False
        self.best_epoch: int = 0

    def step(self, current_metric: float, epoch: int = 0) -> bool:
        """
        Update early stopping counter with current metric value.
        Returns:
            bool: True if training should stop, False otherwise.
        """
        score = current_metric

        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            return False

        if self.mode == "max":
            improved = score > (self.best_score + self.min_delta)
        else:
            improved = score < (self.best_score - self.min_delta)

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def is_improved(self, current_metric: float) -> bool:
        """Check if current metric improved without incrementing counter."""
        if self.best_score is None:
            return True
        if self.mode == "max":
            return current_metric > (self.best_score + self.min_delta)
        else:
            return current_metric < (self.best_score - self.min_delta)

    def state_dict(self) -> Dict[str, Any]:
        """Serialize complete state for checkpointing."""
        return {
            "monitor": self.monitor,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "mode": self.mode,
            "best_score": self.best_score,
            "counter": self.counter,
            "early_stop": self.early_stop,
            "best_epoch": self.best_epoch
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore state from checkpoint."""
        self.monitor = state.get("monitor", self.monitor)
        self.patience = state.get("patience", self.patience)
        self.min_delta = state.get("min_delta", self.min_delta)
        self.mode = state.get("mode", self.mode)
        self.best_score = state.get("best_score", None)
        self.counter = state.get("counter", 0)
        self.early_stop = state.get("early_stop", False)
        self.best_epoch = state.get("best_epoch", 0)
