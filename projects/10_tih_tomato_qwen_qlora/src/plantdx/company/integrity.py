"""Frozen run identity and measured runtime gate for the company trainer."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from plantdx.company.dataset import SPLITS
from plantdx.core.exceptions import PlantDxError


def identity(cfg, dataset, data_dir=None):
    """Bind checkpoints and smoke evidence to data, settings and implementation."""
    directory = Path(data_dir) if data_dir is not None else cfg.dataset_dir
    source = Path(__file__).parent
    from plantdx.company.environment import environment_facts

    facts = environment_facts()
    return {
        "runtime": {
            "packages": facts.packages,
            "gpu": facts.gpu.name,
            "cuda": facts.gpu.cuda_version,
        },
        "run_name": cfg.run_name,
        "seed": cfg.seed,
        "model": asdict(cfg.model),
        "lora": asdict(cfg.lora),
        "optim": asdict(cfg.optim),
        "data": asdict(cfg.data),
        "evaluation": asdict(cfg.evaluation),
        "classes": [asdict(c) for c in dataset.classes],
        "splits": {
            s: hashlib.sha256((directory / f"{s}.jsonl").read_bytes()).hexdigest()
            for s in SPLITS
        },
        "implementation": hashlib.sha256(
            b"".join(p.read_bytes() for p in sorted(source.glob("*.py")))
        ).hexdigest(),
    }


def fingerprint(cfg, dataset):
    """Hash the exact official run identity."""
    return hashlib.sha256(
        json.dumps(identity(cfg, dataset), sort_keys=True).encode()
    ).hexdigest()


def require_budget(cfg, dataset):
    """
    Refuse stale evidence, partial evaluation, or excessively long runs.

    Runtime gate increased for accuracy experiments:
    - Previous limit: 3.5 hours
    - New limit: 12 hours

    This allows higher-resolution QLoRA experiments while
    keeping smoke-test validation requirements intact.
    """

    from plantdx.company.smoke import last_smoke

    result = last_smoke(cfg)

    if cfg.evaluation.limit is not None:
        raise PlantDxError(
            "Official evaluation must cover the full validation and test splits"
        )

    if (
        result is None
        or not result.passed
        or result.fingerprint != fingerprint(cfg, dataset)
        or result.estimated_total_seconds is None
        or result.estimated_total_seconds > 43200
        or not result.resume_verified
        or not result.validation_batch_verified
    ):
        raise PlantDxError(
            "A current passing smoke test with a <=12-hour estimate is required"
        )

    return result