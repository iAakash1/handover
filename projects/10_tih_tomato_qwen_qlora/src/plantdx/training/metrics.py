"""Training metrics logging: CSV + JSON + Markdown (+ optional TensorBoard).

A small, dependency-light logger the runner feeds per-step and per-epoch records
into. CSV/JSON/Markdown are always written; TensorBoard is optional and imported
lazily so its absence never breaks a run or CI. Tracks the best validation loss
so the runner can decide when to keep a "best" checkpoint.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from plantdx.utils.io import ensure_dir


@dataclass
class StepRecord:
    """One logged training step."""

    step: int
    epoch: float
    train_loss: float
    learning_rate: float
    val_loss: float | None = None
    tokens_per_sec: float | None = None


@dataclass
class MetricsLogger:
    """Accumulates step records and writes CSV/JSON/Markdown summaries."""

    log_dir: str | Path
    tensorboard: bool = False
    records: list[StepRecord] = field(default_factory=list)
    best_val_loss: float | None = None
    best_step: int | None = None
    _tb: Any = None

    def __post_init__(self) -> None:
        """Create the log directory and open TensorBoard if requested."""
        ensure_dir(self.log_dir)
        if self.tensorboard:
            self._tb = _open_tensorboard(self.log_dir)

    def log_step(self, record: StepRecord) -> bool:
        """Record a step. Returns True if it set a new best validation loss."""
        self.records.append(record)
        is_best = False
        if record.val_loss is not None and (
            self.best_val_loss is None or record.val_loss < self.best_val_loss
        ):
            self.best_val_loss = record.val_loss
            self.best_step = record.step
            is_best = True
        if self._tb is not None:
            self._tb.add_scalar("train/loss", record.train_loss, record.step)
            self._tb.add_scalar("train/lr", record.learning_rate, record.step)
            if record.val_loss is not None:
                self._tb.add_scalar("val/loss", record.val_loss, record.step)
        return is_best

    def write(self) -> dict[str, str]:
        """Flush CSV, JSON, and Markdown. Returns the written file paths."""
        root = Path(self.log_dir)
        csv_path = root / "metrics.csv"
        json_path = root / "metrics.json"
        md_path = root / "metrics.md"
        self._write_csv(csv_path)
        self._write_json(json_path)
        self._write_markdown(md_path)
        if self._tb is not None:
            self._tb.flush()
        return {"csv": str(csv_path), "json": str(json_path), "markdown": str(md_path)}

    # -- writers ----------------------------------------------------------- #

    def _write_csv(self, path: Path) -> None:
        fields = ["step", "epoch", "train_loss", "learning_rate", "val_loss", "tokens_per_sec"]
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            for rec in self.records:
                writer.writerow(asdict(rec))

    def _write_json(self, path: Path) -> None:
        payload = {
            "best_val_loss": self.best_val_loss,
            "best_step": self.best_step,
            "num_steps": len(self.records),
            "records": [asdict(r) for r in self.records],
        }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def _write_markdown(self, path: Path) -> None:
        lines = ["# Training metrics", ""]
        if self.records:
            first, last = self.records[0], self.records[-1]
            lines += [
                f"- Steps logged: **{len(self.records)}**",
                f"- First train loss: **{first.train_loss:.4f}** (step {first.step})",
                f"- Last train loss: **{last.train_loss:.4f}** (step {last.step})",
            ]
            if self.best_val_loss is not None:
                lines.append(
                    f"- Best val loss: **{self.best_val_loss:.4f}** (step {self.best_step})"
                )
            lines += ["", "| step | epoch | train_loss | lr | val_loss |", "|---|---|---|---|---|"]
            for rec in self.records:
                val = f"{rec.val_loss:.4f}" if rec.val_loss is not None else ""
                lines.append(
                    f"| {rec.step} | {rec.epoch:.2f} | {rec.train_loss:.4f} "
                    f"| {rec.learning_rate:.2e} | {val} |"
                )
        else:
            lines.append("_No steps logged yet._")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _open_tensorboard(log_dir: str | Path) -> Any:
    import importlib

    try:
        module = importlib.import_module("torch.utils.tensorboard")
    except ImportError:
        return None
    return module.SummaryWriter(log_dir=str(log_dir))
