"""Safety properties for company-only training and held-out data."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from plantdx.company.dataset import DatasetError, prepare_dataset, validate_prepared
from plantdx.company.integrity import fingerprint, require_budget
from plantdx.company.modeling import JsonlRows
from plantdx.company.train import balanced_sampler


def test_existing_split_is_immutable(company_config, company_dataset):
    cfg, ds = company_config, company_dataset
    prepare_dataset(cfg, ds)
    before = (cfg.dataset_dir / "test.jsonl").read_bytes()
    prepare_dataset(cfg, ds)
    assert (cfg.dataset_dir / "test.jsonl").read_bytes() == before
    changed = replace(cfg, data=replace(cfg.data, split_seed=8))
    with pytest.raises(DatasetError, match="refusing to reshuffle"):
        prepare_dataset(changed, ds)
    assert (cfg.dataset_dir / "test.jsonl").read_bytes() == before


def test_tampered_image_and_cross_split_duplicate_block(company_config, company_dataset):
    cfg, ds = company_config, company_dataset
    prepare_dataset(cfg, ds)
    train = JsonlRows(cfg.dataset_dir / "train.jsonl").rows
    validation_path = cfg.dataset_dir / "validation.jsonl"
    validation = JsonlRows(validation_path).rows
    validation[0] = train[0]
    validation_path.write_text("\n".join(json.dumps(r) for r in validation), encoding="utf-8")
    with pytest.raises(DatasetError, match="Duplicate"):
        validate_prepared(cfg, ds)


def test_sampler_reproduces_epoch_and_resume_order():
    rows = SimpleNamespace(rows=[{"class_id": "common"}] * 100 + [{"class_id": "rare"}] * 4)

    class Rows:
        def __init__(self):
            self.rows = rows.rows

        def __len__(self):
            return len(self.rows)

    sampler = balanced_sampler(Rows(), 7)
    assert float(sampler.weights[-1] / sampler.weights[0]) == pytest.approx(5)
    sampler.set_epoch(2)
    expected = list(sampler)
    resumed = balanced_sampler(Rows(), 7)
    resumed.set_epoch(2)
    assert list(resumed) == expected
    sampler.set_epoch(3)
    assert list(sampler) != expected
    assert len(expected) == 104


def test_fingerprint_changes_with_resolution(company_config, company_dataset):
    prepare_dataset(company_config, company_dataset)
    old = fingerprint(company_config, company_dataset)
    changed = replace(company_config, model=replace(company_config.model, image_max_side=448))
    assert fingerprint(changed, company_dataset) != old
    with pytest.raises(Exception, match="current passing smoke"):
        require_budget(company_config, company_dataset)


def test_real_early_stopping_callback_stops_when_validation_worsens():
    from transformers import EarlyStoppingCallback, TrainerControl, TrainerState, TrainingArguments

    args = TrainingArguments(
        output_dir="unused",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=[],
        use_cpu=True,
    )
    callback = EarlyStoppingCallback(early_stopping_patience=1)
    state = TrainerState(best_metric=0.5)
    control = TrainerControl()
    callback.on_evaluate(args, state, control, metrics={"eval_loss": 0.4})
    assert not control.should_training_stop
    callback.on_evaluate(args, state, control, metrics={"eval_loss": 0.6})
    assert control.should_training_stop


def test_resume_rejects_changed_lora_before_model_loading(company_config, company_dataset):
    from plantdx.company.integrity import identity
    from plantdx.company.train import TrainingError, run_training

    cfg = company_config
    prepare_dataset(cfg, company_dataset)
    checkpoint = cfg.checkpoint_dir / "checkpoint-1"
    checkpoint.mkdir(parents=True)
    (cfg.checkpoint_dir / "run_identity.json").write_text(
        json.dumps(identity(cfg, company_dataset)), encoding="utf-8"
    )
    changed = replace(cfg, lora=replace(cfg.lora, rank=32))
    with pytest.raises(TrainingError, match="Resume rejected"):
        run_training(changed, company_dataset, max_steps=2, resume=True)


def test_final_test_is_full_cached_and_bound_to_adapter(
    company_config, company_dataset, monkeypatch
):
    from plantdx.company import evaluate

    cfg, ds = company_config, company_dataset
    prepare_dataset(cfg, ds)
    cfg.checkpoint_dir.mkdir(parents=True)
    (cfg.checkpoint_dir / "best_checkpoint.json").write_text("{}")
    weights = cfg.checkpoint_dir / "adapter_model.safetensors"
    weights.write_bytes(b"selected adapter")
    labels = {r["image"]: r["answer"] for r in JsonlRows(cfg.dataset_dir / "test.jsonl").rows}
    calls = []

    def generate(_loaded, image, *_args, **_kwargs):
        calls.append(image)
        return labels[image]

    monkeypatch.setattr(evaluate, "generate_label", generate)
    result = evaluate.run_evaluation(
        cfg, ds, split="test", adapter_dir=cfg.checkpoint_dir, loaded=object()
    )
    assert result.samples == len(labels)
    assert result.f1_macro == 1.0
    assert (cfg.report_dir / "test_finetuned" / "confusion_matrix.json").is_file()
    evaluate.run_evaluation(cfg, ds, split="test", adapter_dir=cfg.checkpoint_dir, loaded=object())
    assert len(calls) == len(labels)
    weights.write_bytes(b"different adapter")
    with pytest.raises(evaluate.EvaluationError, match="already used"):
        evaluate.run_evaluation(
            cfg, ds, split="test", adapter_dir=cfg.checkpoint_dir, loaded=object()
        )
