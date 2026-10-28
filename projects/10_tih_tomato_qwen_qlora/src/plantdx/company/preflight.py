"""Preflight checks for the company training run.

Every check reports rather than raises, so the caller can show the whole list at
once. A check is ``critical`` when training cannot possibly succeed without it;
those gate the Train button. Warnings (class imbalance, for instance) are
surfaced but never block.
"""

from __future__ import annotations

import importlib.util
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from plantdx.company.classes import CompanyDataset, load_company_dataset
from plantdx.company.config import CompanyTrainConfig, load_company_train_config
from plantdx.company.dataset import FORBIDDEN_FRAGMENTS, SPLITS
from plantdx.company.environment import environment_facts, free_disk_gb

OK = "ok"
WARN = "warn"
FAIL = "fail"

MIN_FREE_DISK_GB = 10.0
MIN_VRAM_GB = 23.0


@dataclass(frozen=True)
class Check:
    """One preflight result."""

    name: str
    status: str
    detail: str
    critical: bool = True

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready mapping."""
        return asdict(self)


@dataclass
class PreflightReport:
    """The full preflight outcome."""

    checks: list[Check]

    @property
    def ready(self) -> bool:
        """Whether every critical check passed."""
        return not any(c.status == FAIL and c.critical for c in self.checks)

    @property
    def blocking(self) -> list[Check]:
        """Critical checks that failed."""
        return [c for c in self.checks if c.status == FAIL and c.critical]

    @property
    def warnings(self) -> list[Check]:
        """Non-blocking concerns."""
        return [c for c in self.checks if c.status == WARN or (c.status == FAIL and not c.critical)]

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready mapping."""
        return {"ready": self.ready, "checks": [c.to_dict() for c in self.checks]}


def run_preflight(
    *,
    config_path: str | Path | None = None,
    profile_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
    deep: bool = False,
    local_files_only: bool = True,
) -> PreflightReport:
    """Run every check and return the report.

    Args:
        config_path: Training config to validate.
        profile_path: Company dataset profile to validate.
        overrides: Advanced-settings overrides applied before validation.
        deep: Also load the processor and push one real sample through it. Slower
            (tens of seconds) but proves the formatting path end to end.
        local_files_only: Require the base model to already be cached locally.
    """
    checks: list[Check] = []
    cfg: CompanyTrainConfig | None = None
    dataset: CompanyDataset | None = None

    checks.extend(_check_packages())
    checks.extend(_check_gpu())

    try:
        cfg = load_company_train_config(config_path, overrides=overrides)
        checks.append(Check("Training config", OK, f"{cfg.config_path} validates"))
    except Exception as exc:
        checks.append(Check("Training config", FAIL, str(exc)))

    try:
        profile = profile_path or (cfg.data.profile if cfg else None)
        dataset = load_company_dataset(profile)
        checks.append(
            Check(
                "Company dataset",
                OK,
                f"{dataset.name} at {dataset.root} ({len(dataset.classes)} classes mapped)",
            )
        )
    except Exception as exc:
        checks.append(Check("Company dataset", FAIL, str(exc)))

    if dataset is not None:
        checks.extend(_check_dataset(dataset))
    if cfg is not None:
        checks.extend(_check_outputs(cfg))
        checks.extend(_check_prepared(cfg, dataset))
        try:
            from plantdx.company.dataset import validate_prepared

            validate_prepared(cfg, dataset)
            checks.append(
                Check(
                    "Frozen split integrity",
                    OK,
                    "hashes, counts, labels and company provenance verified",
                )
            )
        except Exception as exc:
            checks.append(Check("Frozen split integrity", FAIL, str(exc)))
        checks.append(_check_model_files(cfg, local_files_only=local_files_only))
        checks.append(_check_smoke(cfg))
        checks.append(_check_evaluation_pathway())
        if deep:
            checks.extend(_check_sample(cfg, dataset))

    return PreflightReport(checks=checks)


def _check_smoke(cfg: CompanyTrainConfig) -> Check:
    """Whether the tiny real QLoRA run has been proved on this machine.

    Reads the recorded result; it never trains. Running the smoke test is an
    explicit action, so preflight can report it without ever starting work.
    """
    from plantdx.company.smoke import last_smoke

    try:
        result = last_smoke(cfg)
    except Exception as exc:
        return Check("QLoRA smoke test", FAIL, f"could not read the smoke result: {exc}")
    if result is None:
        return Check(
            "QLoRA smoke test",
            FAIL,
            "not run yet - run it from the Training page (or `plantdx company smoke`) "
            "to prove forward/backward/optimizer/adapter-save on this GPU",
        )
    if not result.passed:
        return Check("QLoRA smoke test", FAIL, result.detail)
    try:
        from plantdx.company.integrity import require_budget

        require_budget(cfg, load_company_dataset(cfg.data.profile))
        if not result.resume_verified or not result.validation_batch_verified:
            raise ValueError("Resume and validation batch verification required")
    except Exception as exc:
        return Check("QLoRA smoke test / runtime", FAIL, str(exc))
    loss = f", final loss {result.final_loss:.3f}" if result.final_loss is not None else ""
    memory = f", peak {result.peak_memory_gb} GB" if result.peak_memory_gb else ""
    return Check("QLoRA smoke test", OK, f"passed - {result.detail}{loss}{memory}")


def _check_evaluation_pathway() -> Check:
    """Whether the post-training evaluation can run (metrics stack + parser)."""
    if importlib.util.find_spec("sklearn") is None:
        return Check(
            "Evaluation pathway",
            FAIL,
            "scikit-learn is missing, so post-training metrics cannot be computed",
            critical=True,
        )
    try:
        from pathlib import Path as _Path

        from plantdx.company.classes import ClassEntry, CompanyDataset
        from plantdx.company.evaluate import UNPARSEABLE, parse_prediction

        probe = CompanyDataset(
            name="probe",
            crop="tomato",
            root=_Path(),
            extensions=(".jpg",),
            classes=(
                ClassEntry(
                    raw="Late_Blight",
                    class_id="late_blight",
                    label="Late Blight",
                ),
            ),
            profile_path=_Path(),
        )
        if parse_prediction("Diagnosis: Late Blight", probe) != "late_blight":
            return Check(
                "Evaluation pathway", FAIL, "the answer parser did not resolve a known label"
            )
        if parse_prediction("something else entirely", probe) != UNPARSEABLE:
            return Check(
                "Evaluation pathway", FAIL, "the answer parser guessed at an unknown answer"
            )
    except Exception as exc:
        return Check("Evaluation pathway", FAIL, f"{type(exc).__name__}: {exc}")
    return Check("Evaluation pathway", OK, "metrics stack importable, answer parser verified")


def _check_packages() -> list[Check]:
    facts = environment_facts()
    checks: list[Check] = []
    required = ("torch", "transformers", "peft", "accelerate", "bitsandbytes")
    missing = [name for name in required if facts.packages.get(name) == "not installed"]
    if missing:
        checks.append(
            Check(
                "Python packages",
                FAIL,
                "not installed: "
                + ", ".join(missing)
                + ". Install with: pip install torch transformers peft accelerate bitsandbytes",
            )
        )
    else:
        checks.append(
            Check(
                "Python packages",
                OK,
                ", ".join(f"{k} {v}" for k, v in facts.packages.items() if k in required),
            )
        )

    if importlib.util.find_spec("sklearn") is None:
        checks.append(
            Check(
                "scikit-learn",
                FAIL,
                "not installed; evaluation metrics need it (pip install scikit-learn)",
                critical=True,
            )
        )
    else:
        checks.append(Check("scikit-learn", OK, "available for evaluation metrics"))
    return checks


def _check_gpu() -> list[Check]:
    facts = environment_facts()
    gpu = facts.gpu
    if not gpu.available:
        return [
            Check(
                "CUDA GPU",
                FAIL,
                f"{gpu.reason}. This training path needs an NVIDIA GPU with CUDA.",
            )
        ]
    checks = [
        Check(
            "CUDA GPU",
            OK,
            f"{gpu.name} | {gpu.total_vram_gb} GB | CUDA {gpu.cuda_version} | "
            f"compute {gpu.capability}",
        )
    ]
    if gpu.total_vram_gb is not None and gpu.total_vram_gb < MIN_VRAM_GB:
        checks.append(
            Check(
                "VRAM",
                FAIL,
                f"{gpu.total_vram_gb} GB is below the {MIN_VRAM_GB} GB this run expects; "
                "lower model.image_max_side if training runs out of memory",
                critical=True,
            )
        )
    if "RTX 4500 Ada" not in (gpu.name or ""):
        checks.append(Check("Target GPU", FAIL, f"Expected RTX 4500 Ada; found {gpu.name}"))
    if not gpu.bf16:
        checks.append(
            Check(
                "BF16",
                FAIL,
                "this GPU does not report bf16 support; set model.compute_dtype to float16",
                critical=True,
            )
        )
    else:
        checks.append(Check("BF16", OK, "supported"))
    return checks


def _check_dataset(dataset: CompanyDataset) -> list[Check]:
    checks: list[Check] = []
    missing = [c.raw for c in dataset.classes if not (dataset.root / c.raw).is_dir()]
    if missing:
        checks.append(
            Check("Class folders", FAIL, "missing folder(s): " + ", ".join(sorted(missing)))
        )
    else:
        checks.append(
            Check(
                "Class folders",
                OK if len(dataset.classes) == 11 else FAIL,
                f"all {len(dataset.classes)} present; expected 11",
            )
        )

    unmapped = dataset.unmapped_dirs()
    if unmapped:
        checks.append(
            Check(
                "Unmapped folders",
                FAIL,
                "excluded from training (add them to the profile to include): "
                + ", ".join(unmapped),
                critical=True,
            )
        )
    else:
        checks.append(Check("Unmapped folders", OK, "every folder has a class mapping"))

    allowed = set(dataset.extensions)
    counts: dict[str, int] = {}
    other_ext = 0
    for entry in dataset.classes:
        class_dir = dataset.root / entry.raw
        if not class_dir.is_dir():
            continue
        found = 0
        for path in class_dir.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() in allowed:
                found += 1
            else:
                other_ext += 1
        counts[entry.class_id] = found

    total = sum(counts.values())
    empty = sorted(k for k, v in counts.items() if v == 0)
    if total == 0:
        checks.append(Check("Images", FAIL, f"no supported images under {dataset.root}"))
    elif empty:
        checks.append(
            Check("Images", FAIL, f"{total} images, but empty class(es): " + ", ".join(empty))
        )
    else:
        checks.append(Check("Images", OK, f"{total} images across {len(counts)} classes"))

    if other_ext:
        checks.append(
            Check(
                "Image formats",
                WARN,
                f"{other_ext} file(s) with an unsupported extension will be skipped",
                critical=False,
            )
        )
    else:
        checks.append(Check("Image formats", OK, "every file has a supported extension"))

    if counts:
        values = [v for v in counts.values() if v]
        ratio = round(max(values) / min(values), 1) if values else 0.0
        if ratio >= 5:
            checks.append(
                Check(
                    "Class balance",
                    WARN,
                    f"largest class is {ratio}x the smallest; expect weaker recall on rare classes",
                    critical=False,
                )
            )
        else:
            checks.append(Check("Class balance", OK, f"max/min class ratio {ratio}x"))

    lowered = str(dataset.root).replace("\\", "/").lower()
    leaked = [f for f in FORBIDDEN_FRAGMENTS if f in lowered]
    if leaked:
        checks.append(
            Check("Dataset isolation", FAIL, f"dataset root contains {leaked[0]!r}: {dataset.root}")
        )
    else:
        checks.append(
            Check("Dataset isolation", OK, "root is company-only (no PlantVillage, no mango)")
        )
    return checks


def _check_outputs(cfg: CompanyTrainConfig) -> list[Check]:
    checks: list[Check] = []
    problems: list[str] = []
    for label, path in (
        ("dataset", cfg.dataset_dir),
        ("checkpoints", cfg.checkpoint_dir),
        ("logs", cfg.log_dir),
        ("reports", cfg.report_dir),
    ):
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / ".write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
        except OSError as exc:
            problems.append(f"{label}: {exc}")
    if problems:
        checks.append(Check("Output directories", FAIL, "; ".join(problems)))
    else:
        checks.append(Check("Output directories", OK, f"writable under {cfg.base_dir}"))

    free = free_disk_gb(cfg.checkpoint_dir)
    if free is None:
        checks.append(Check("Disk space", WARN, "could not read free space", critical=False))
    elif free < MIN_FREE_DISK_GB:
        checks.append(
            Check("Disk space", FAIL, f"{free} GB free; need at least {MIN_FREE_DISK_GB} GB")
        )
    else:
        checks.append(Check("Disk space", OK, f"{free} GB free"))
    return checks


def _check_prepared(cfg: CompanyTrainConfig, dataset: CompanyDataset | None) -> list[Check]:
    """Validate an already-prepared dataset; absence is a warning, not a failure."""
    import json

    data_dir = cfg.dataset_dir
    missing = [s for s in SPLITS if not (data_dir / f"{s}.jsonl").is_file()]
    if missing:
        return [
            Check(
                "Prepared dataset",
                WARN,
                "not built yet; it will be prepared before training "
                f"(missing: {', '.join(missing)})",
                critical=False,
            )
        ]

    checks: list[Check] = []
    counts: dict[str, int] = {}
    bad_paths: list[str] = []
    leaked: list[str] = []
    labels = set(dataset.labels) if dataset else set()
    unknown_labels: set[str] = set()

    for split in SPLITS:
        path = data_dir / f"{split}.jsonl"
        rows = 0
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rows += 1
                try:
                    row = json.loads(line)
                except ValueError as exc:
                    bad_paths.append(f"{split}:{rows} unparseable JSON ({exc})")
                    continue
                image = str(row.get("image", ""))
                lowered = image.replace("\\", "/").lower()
                for fragment in FORBIDDEN_FRAGMENTS:
                    if fragment in lowered:
                        leaked.append(image)
                        break
                if len(bad_paths) < 5 and not Path(image).is_file():
                    bad_paths.append(f"{split}: missing image {image}")
                if labels and str(row.get("answer", "")) not in labels:
                    unknown_labels.add(str(row.get("answer", "")))
        counts[split] = rows

    checks.append(
        Check(
            "Prepared dataset",
            OK if not bad_paths else FAIL,
            (
                "train {train} | validation {validation} | test {test}".format(**counts)
                if not bad_paths
                else "; ".join(bad_paths[:5])
            ),
        )
    )
    checks.append(
        Check(
            "Training-set provenance",
            FAIL if leaked else OK,
            (
                f"{len(leaked)} row(s) point outside the company dataset: {leaked[0]}"
                if leaked
                else "no PlantVillage or mango path in any split"
            ),
        )
    )
    if unknown_labels:
        checks.append(
            Check(
                "Label vocabulary",
                FAIL,
                "answer(s) not in the class profile: " + ", ".join(sorted(unknown_labels)[:5]),
            )
        )
    elif labels:
        checks.append(
            Check("Label vocabulary", OK, f"every answer is one of {len(labels)} classes")
        )
    return checks


def _check_model_files(cfg: CompanyTrainConfig, *, local_files_only: bool) -> Check:
    """Whether the base model can be resolved without a download."""
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(cfg.model.model_id, local_files_only=True, allow_patterns=["*.json"])
        return Check("Base model", OK, f"{cfg.model.model_id} present in the local cache")
    except Exception as exc:
        status = FAIL if local_files_only else WARN
        return Check(
            "Base model",
            status,
            f"{cfg.model.model_id} is not fully cached ({type(exc).__name__}); "
            "it will be downloaded on first use",
            critical=local_files_only,
        )


def _check_sample(cfg: CompanyTrainConfig, dataset: CompanyDataset | None) -> list[Check]:
    """Push one real row through the processor and the collator."""
    from plantdx.company.modeling import CompanyCollator, JsonlRows, load_processor

    try:
        processor = load_processor(cfg, local_files_only=True)
    except Exception as exc:
        return [Check("Processor", FAIL, str(exc))]

    checks = [Check("Processor", OK, type(processor).__name__)]
    train_file = cfg.dataset_dir / "validation.jsonl"
    row: dict[str, Any] | None = None
    if train_file.is_file():
        try:
            row = JsonlRows(train_file, limit=1).rows[0]
        except Exception as exc:
            return [*checks, Check("Sample batch", FAIL, str(exc))]
    elif dataset is not None:
        for entry in dataset.classes:
            class_dir = dataset.root / entry.raw
            if not class_dir.is_dir():
                continue
            found = next(
                (
                    p
                    for p in sorted(class_dir.rglob("*"))
                    if p.is_file() and p.suffix.lower() in set(dataset.extensions)
                ),
                None,
            )
            if found is not None:
                row = {
                    "image": str(found),
                    "question": cfg.data.instruction,
                    "answer": entry.label,
                }
                break
    if row is None:
        return [*checks, Check("Sample batch", FAIL, "no image available to test")]

    try:
        collator = CompanyCollator(
            processor,
            max_side=cfg.model.image_max_side,
            max_seq_length=cfg.model.max_seq_length,
        )
        batch = collator([row])
        supervised = int((batch["labels"] != -100).sum())
        checks.append(
            Check(
                "Sample batch",
                OK if supervised else FAIL,
                f"{batch['input_ids'].shape[1]} tokens, {supervised} supervised, "
                f"pixels {tuple(batch['pixel_values'].shape)}",
            )
        )
    except Exception as exc:
        checks.append(Check("Sample batch", FAIL, f"{type(exc).__name__}: {exc}"))
    return checks
