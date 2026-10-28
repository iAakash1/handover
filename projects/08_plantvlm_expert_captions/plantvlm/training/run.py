"""
Run isolation and lineage — the reproducibility layer.

WHY THIS EXISTS
    In the previous design every run wrote to the same outputs/ directory. The
    second run silently destroyed the first, comparing two configurations meant
    renaming folders by hand, and a checkpoint carried no record of the data or
    settings that produced it. That is fine for one experiment and fatal for a
    research project.

    A run is now a directory, created once, never overwritten:

        runs/run_003_fold0_20260805-142233/
        ├── config.yaml        the exact settings, frozen at launch
        ├── lineage.json       source workbook hash, per-image hashes, split
        │                      membership, environment versions
        ├── data/              the exact JSONL this run trained on
        ├── checkpoints/       epoch-N/ and best/
        ├── logs/              train.jsonl, epochs.jsonl
        ├── predictions/       written by the inference stage
        └── evaluation/        written by the evaluation stage

    Two properties follow, and both are what a paper needs. Any number in a
    report can be traced to the data and settings that produced it. And nothing
    downstream ever has to recompute a metric to redraw a figure — the artifact
    is already on disk.

INPUT   a Config, plus the dataset manifest
OUTPUT  a directory that fully describes one experiment
"""

from __future__ import annotations

import platform
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from ..config import Config
from ..io import read_json, sha256_file, utc_now, write_json


def _versions() -> Dict[str, str]:
    """Record what the code ran against — version drift is a real failure mode."""
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for name in ("torch", "transformers", "peft", "accelerate"):
        try:
            out[name] = __import__(name).__version__
        except Exception:
            out[name] = "not installed"
    try:
        import torch
        if torch.cuda.is_available():
            out["gpu"] = torch.cuda.get_device_name(0)
            out["cuda"] = torch.version.cuda or "unknown"
    except Exception:
        pass
    return out


@dataclass
class Run:
    """A single experiment on disk."""

    directory: Path

    # ── Paths ────────────────────────────────────────────────────────────────

    @property
    def config_file(self) -> Path: return self.directory / "config.yaml"

    @property
    def lineage_file(self) -> Path: return self.directory / "lineage.json"

    @property
    def data_dir(self) -> Path: return self.directory / "data"

    @property
    def checkpoints(self) -> Path: return self.directory / "checkpoints"

    @property
    def best(self) -> Path: return self.checkpoints / "best"

    @property
    def logs(self) -> Path: return self.directory / "logs"

    @property
    def predictions(self) -> Path: return self.directory / "predictions"

    @property
    def evaluation(self) -> Path: return self.directory / "evaluation"

    def checkpoint(self, epoch: int) -> Path:
        return self.checkpoints / f"epoch-{epoch}"

    def split_file(self, split: str) -> Path:
        return self.data_dir / f"{split}.jsonl"

    @property
    def name(self) -> str: return self.directory.name

    # ── Lifecycle ────────────────────────────────────────────────────────────

    @classmethod
    def create(cls, cfg: Config, tag: str = "") -> "Run":
        """Allocate the next run directory, freeze the config, copy the data."""
        cfg.paths.runs_dir.mkdir(parents=True, exist_ok=True)
        existing = [p.name for p in cfg.paths.runs_dir.iterdir() if p.is_dir()]
        index = len([n for n in existing if n.startswith("run_")]) + 1
        suffix = f"_{tag}" if tag else ""
        name = f"run_{index:03d}_fold{cfg.split.fold}{suffix}"

        run = cls(cfg.paths.runs_dir / name)
        for d in (run.directory, run.data_dir, run.checkpoints, run.logs,
                  run.predictions, run.evaluation):
            d.mkdir(parents=True, exist_ok=True)

        cfg.freeze_to(run.config_file)

        for split in ("train", "val", "test"):
            source = cfg.paths.split_file(split)
            if source.exists():
                shutil.copy2(source, run.split_file(split))

        (cfg.paths.runs_dir / "latest.txt").write_text(name, encoding="utf-8")
        return run

    @classmethod
    def latest(cls, cfg: Config) -> "Run":
        """The most recently created run — the default target for later stages."""
        pointer = cfg.paths.runs_dir / "latest.txt"
        if not pointer.exists():
            raise FileNotFoundError(
                f"no runs yet in {cfg.paths.runs_dir} — train something first")
        return cls(cfg.paths.runs_dir / pointer.read_text(encoding="utf-8").strip())

    @classmethod
    def resolve(cls, cfg: Config, name: Optional[str]) -> "Run":
        """A named run, or the latest when name is None."""
        if not name:
            return cls.latest(cfg)
        directory = Path(name)
        if not directory.is_absolute():
            directory = cfg.paths.runs_dir / name
        if not directory.exists():
            raise FileNotFoundError(f"run not found: {directory}")
        return cls(directory)

    # ── Lineage ──────────────────────────────────────────────────────────────

    def write_lineage(self, cfg: Config, manifest: Dict,
                      extra: Optional[Dict] = None) -> None:
        """
        Everything needed to answer "what exactly produced this?" — later, by
        someone else, without asking.
        """
        code_dir = Path(__file__).resolve().parent.parent
        code_hashes = {
            p.relative_to(code_dir).as_posix(): sha256_file(p)[:16]
            for p in sorted(code_dir.rglob("*.py"))
        }
        write_json({
            "run": self.name,
            "created": utc_now(),
            "config_file": cfg.source_file,
            "instruction": cfg.data.instruction,
            "source": manifest.get("source"),
            "split": manifest.get("split"),
            "n_images": manifest.get("n_images"),
            "n_pairs": manifest.get("n_pairs"),
            "image_hashes": {k: v["sha256"] for k, v in
                             manifest.get("images", {}).items()},
            "environment": _versions(),
            "code_sha256": code_hashes,
            **(extra or {}),
        }, self.lineage_file)

    def read_lineage(self) -> Dict:
        return read_json(self.lineage_file)

    # ── Reporting ────────────────────────────────────────────────────────────

    def summary(self) -> str:
        lines = [f"  run       : {self.name}", f"  directory : {self.directory}"]
        epochs = self.logs / "epochs.jsonl"
        if epochs.exists():
            rows = [line for line in epochs.read_text(encoding="utf-8").splitlines()
                    if line.strip()]
            lines.append(f"  epochs    : {len(rows)} logged")
        if self.best.exists():
            lines.append(f"  best      : {self.best}")
        return "\n".join(lines)
