"""Synthetic fixtures for training-pipeline tests (no real images/corpus needed).

Everything here is generated into ``tmp_path`` so the tests are fast, hermetic,
and independent of the gitignored ``datasets/`` and ``artifacts/corpus/`` trees.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from plantdx.training.config import (
    CheckpointConfig,
    DataConfig,
    LoggingConfig,
    LoRAConfig,
    ModelConfig,
    OptimConfig,
    TrainingConfig,
)
from plantdx.utils.io import write_jsonl

TOMATO_CLASSES = (
    "healthy",
    "bacterial_spot",
    "early_blight",
    "late_blight",
    "leaf_mold",
    "septoria_leaf_spot",
    "spider_mites",
    "target_spot",
    "mosaic_virus",
    "yellow_leaf_curl_virus",
)


@pytest.fixture
def processed_tree(tmp_path: Path) -> Path:
    """A ``datasets/`` root with 6 fake tomato images per class."""
    root = tmp_path / "datasets" / "tomato" / "processed"
    for cls in TOMATO_CLASSES:
        d = root / cls
        d.mkdir(parents=True)
        for i in range(6):
            (d / f"{cls}_{i:03d}.JPG").write_bytes(b"")
    return tmp_path / "datasets"


@pytest.fixture
def corpus_file(tmp_path: Path) -> Path:
    """A synthetic corpus JSONL + checksum.txt with 4 captions per tomato disease."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    rows = []
    for cls in TOMATO_CLASSES:
        disease = f"tomato_{cls}"
        for j in range(4):
            rows.append(
                {
                    "caption_id": f"cap_{disease}_{j}",
                    "condition": disease,
                    "crop": "tomato",
                    "text": f"This tomato leaf shows {cls.replace('_', ' ')} sign {j}.",
                }
            )
    write_jsonl(corpus_dir / "captions.jsonl", rows)
    (corpus_dir / "checksum.txt").write_text("sha256:deadbeef\n", encoding="utf-8")
    return corpus_dir / "captions.jsonl"


@pytest.fixture
def data_config(processed_tree: Path, corpus_file: Path) -> DataConfig:
    """A tomato DataConfig pointed at the synthetic tree + corpus."""
    return DataConfig(
        crop="tomato",
        classes=TOMATO_CLASSES,
        processed_dir=str(processed_tree),
        corpus_path=str(corpus_file),
        captions_per_image=2,
        max_captions_per_disease=200,
        response_format="qa",
        train_ratio=0.5,
        val_ratio=0.25,
        test_ratio=0.25,
        split_seed=20260711,
        image_glob="*.JPG",
    )


@pytest.fixture
def training_config(data_config: DataConfig, tmp_path: Path) -> TrainingConfig:
    """A complete TrainingConfig with outputs redirected under tmp_path."""
    return TrainingConfig(
        run_name="test_run",
        backend="mlx_vlm",
        seed=20260711,
        grad_checkpoint=True,
        model=ModelConfig(
            name="qwen2_5_vl",
            model_path="mlx-community/Qwen2.5-VL-7B-Instruct-4bit",
            family="qwen2_5_vl",
            max_seq_length=2048,
            quantization_bits=4,
            image_resize=448,
        ),
        lora=LoRAConfig(
            method="qlora", rank=16, alpha=32, dropout=0.05, target_modules=("q_proj",)
        ),
        optim=OptimConfig(
            learning_rate=1e-4,
            min_learning_rate=1e-6,
            warmup_steps=5,
            scheduler="cosine",
            grad_clip=1.0,
            gradient_accumulation_steps=8,
            batch_size=1,
            epochs=1,
        ),
        data=data_config,
        checkpoint=CheckpointConfig(
            output_dir=str(tmp_path / "checkpoints" / "test_run"),
            steps_per_save=10,
            keep_last=3,
            keep_best=True,
            resume=False,
        ),
        logging=LoggingConfig(
            log_dir=str(tmp_path / "logs" / "test_run"),
            report_dir=str(tmp_path / "reports" / "test_run"),
            steps_per_report=10,
            steps_per_eval=200,
            val_batches=4,
            tensorboard=False,
        ),
    )


@pytest.fixture
def inline_cfg(training_config: TrainingConfig, tmp_path: Path) -> Path:
    """Serialize the fixture TrainingConfig into an inline (uncomposed) YAML file."""
    import yaml

    m, lo, op, d = (
        training_config.model,
        training_config.lora,
        training_config.optim,
        training_config.data,
    )
    raw = {
        "run_name": training_config.run_name,
        "backend": "mlx_vlm",
        "seed": training_config.seed,
        "grad_checkpoint": training_config.grad_checkpoint,
        "model": {
            "name": m.name,
            "model_path": m.model_path,
            "family": m.family,
            "max_seq_length": m.max_seq_length,
            "quantization_bits": m.quantization_bits,
            "image_resize": m.image_resize,
        },
        "lora": {
            "method": lo.method,
            "rank": lo.rank,
            "alpha": lo.alpha,
            "dropout": lo.dropout,
            "target_modules": list(lo.target_modules),
        },
        "optim": {
            "learning_rate": op.learning_rate,
            "min_learning_rate": op.min_learning_rate,
            "warmup_steps": op.warmup_steps,
            "scheduler": op.scheduler,
            "grad_clip": op.grad_clip,
            "gradient_accumulation_steps": op.gradient_accumulation_steps,
            "batch_size": op.batch_size,
            "epochs": op.epochs,
        },
        "data": {
            "crop": d.crop,
            "classes": list(d.classes),
            "processed_dir": d.processed_dir,
            "corpus_path": d.corpus_path,
            "captions_per_image": d.captions_per_image,
            "max_captions_per_disease": d.max_captions_per_disease,
            "response_format": d.response_format,
            "train_ratio": d.train_ratio,
            "val_ratio": d.val_ratio,
            "test_ratio": d.test_ratio,
            "split_seed": d.split_seed,
        },
        "checkpoint": {
            "output_dir": training_config.checkpoint.output_dir,
            "steps_per_save": training_config.checkpoint.steps_per_save,
        },
        "logging": {
            "log_dir": training_config.logging.log_dir,
            "report_dir": training_config.logging.report_dir,
            "steps_per_report": training_config.logging.steps_per_report,
            "steps_per_eval": training_config.logging.steps_per_eval,
        },
    }
    path = tmp_path / "inline.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path
