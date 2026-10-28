"""Stage 1 orchestration: adapter path resolution, without needing real mlx-vlm.

``load_model``/``_generate`` are mocked (real mlx-vlm inference needs Apple
Silicon + a loaded 7B model; that path is exercised manually, not in CI) --
these tests instead pin down the *contract* between ``run_inference`` and the
checkpoint resolver, which is exactly where the reported bug lived.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from plantdx.evaluation.checkpoint import ADAPTER_CONFIG_NAME, ADAPTER_WEIGHTS_NAME
from plantdx.evaluation.config import resolve_eval_config
from plantdx.training.inference import LoadedModel


@pytest.fixture
def checkpoint_dir(tmp_path: Path) -> Path:
    ckpt = tmp_path / "checkpoints" / "test_run"
    ckpt.mkdir(parents=True)
    (ckpt / ADAPTER_CONFIG_NAME).write_text("{}", encoding="utf-8")
    (ckpt / ADAPTER_WEIGHTS_NAME).write_bytes(b"fake-weights")
    return ckpt


@pytest.fixture
def eval_dataset_dir(tmp_path: Path) -> Path:
    d = tmp_path / "dataset"
    d.mkdir()
    row = {
        "image": "/data/tomato/processed/healthy/img.JPG",
        "question": "Describe the visible condition of this tomato leaf.",
        "answer": "This is a healthy tomato leaf.",
    }
    (d / "train.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    (d / "test.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    # Real dataset dirs always carry a manifest (written by the training data
    # pipeline); `run_inference` reads `crop` from it -- see
    # `evaluation/config.py::resolve_crop`.
    (d / "manifest.json").write_text(json.dumps({"crop": "tomato"}), encoding="utf-8")
    return d


def _fake_loaded_model(model_path: str, adapter_path: str | None = None) -> LoadedModel:
    return LoadedModel(
        model=object(),
        processor=object(),
        config=object(),
        model_path=model_path,
        adapter_path=adapter_path,
    )


@pytest.mark.unit
def test_run_inference_passes_the_resolved_directory_not_the_raw_config(
    checkpoint_dir: Path, eval_dataset_dir: Path, tmp_path: Path
) -> None:
    """Regression test for the reported bug: passing a path ending in
    `adapters.safetensors` in `--adapter` must still reach `load_model` as the
    checkpoint DIRECTORY, never as the raw file path."""
    from plantdx.evaluation import inference_runner

    weights_file_path = str(checkpoint_dir / ADAPTER_WEIGHTS_NAME)  # the exact buggy input
    cfg = resolve_eval_config(
        stage="inference",
        adapter=weights_file_path,
        dataset=str(eval_dataset_dir),
        output_dir=str(tmp_path / "out"),
    )

    captured_adapter_paths: list[str | None] = []

    def fake_load_model(model_path: str, adapter_path: str | None = None) -> LoadedModel:
        captured_adapter_paths.append(adapter_path)
        return _fake_loaded_model(model_path, adapter_path)

    with (
        patch.object(inference_runner, "load_model", side_effect=fake_load_model),
        patch.object(
            inference_runner,
            "_generate",
            return_value=inference_runner._GenerationTelemetry(
                text="ok",
                elapsed_ms=1.0,
                prompt_tokens=1,
                generation_tokens=1,
                peak_memory_gb=0.0,
                confidence=None,
            ),
        ),
    ):
        inference_runner.run_inference(cfg)

    # base model: no adapter; fine-tuned model: the RESOLVED directory, never
    # the raw `.../adapters.safetensors` file path from the config.
    assert captured_adapter_paths == [None, str(checkpoint_dir)]
    assert weights_file_path not in captured_adapter_paths


@pytest.mark.unit
def test_run_inference_writes_predictions_and_metadata(
    checkpoint_dir: Path, eval_dataset_dir: Path, tmp_path: Path
) -> None:
    from plantdx.evaluation import inference_runner

    cfg = resolve_eval_config(
        stage="inference",
        adapter=str(checkpoint_dir),
        dataset=str(eval_dataset_dir),
        output_dir=str(tmp_path / "out"),
    )

    with (
        patch.object(inference_runner, "load_model", side_effect=_fake_loaded_model),
        patch.object(
            inference_runner,
            "_generate",
            return_value=inference_runner._GenerationTelemetry(
                text="a healthy tomato leaf",
                elapsed_ms=5.0,
                prompt_tokens=10,
                generation_tokens=5,
                peak_memory_gb=8.0,
                confidence=0.9,
            ),
        ),
    ):
        out_path = inference_runner.run_inference(cfg)

    assert out_path.is_file()
    rows = [json.loads(line) for line in out_path.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["finetuned_prediction"] == "a healthy tomato leaf"
    metadata = json.loads((out_path.parent / "metadata.json").read_text())
    assert metadata["sample_count"] == 1
    assert metadata["crop"] == "tomato"


@pytest.fixture
def mango_eval_dataset_dir(tmp_path: Path) -> Path:
    """A mango-crop dataset dir, mirroring the shape `plantdx train --crop
    mango` actually produces. Used for the regression test below: real image
    disease IDs must be recovered from the DATASET's own declared crop, never
    a hardcoded label map."""
    d = tmp_path / "mango_dataset"
    d.mkdir()
    row = {
        "image": "/data/mango/processed/anthracnose/img.jpg",
        "question": "Describe what is shown on this mango leaf image.",
        "answer": "This mango leaf is affected by anthracnose.",
    }
    (d / "train.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    (d / "test.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    (d / "manifest.json").write_text(json.dumps({"crop": "mango"}), encoding="utf-8")
    return d


@pytest.mark.unit
@pytest.mark.requires_dkb
def test_run_inference_resolves_mango_disease_ids_not_unknown_or_tomato(
    checkpoint_dir: Path, mango_eval_dataset_dir: Path, tmp_path: Path
) -> None:
    """Regression test for the reported bug: a mango image's ground-truth
    `disease_id` must resolve to `mango_anthracnose` via the real, committed
    `assets/metadata/label_map.json` -- never `unknown:anthracnose` (the old
    hardcoded `load_label_map("tomato")` had no "anthracnose" key at all) and
    never a wrong-crop id from a folder-name collision (e.g. "healthy")."""
    from plantdx.evaluation import inference_runner

    cfg = resolve_eval_config(
        stage="inference",
        adapter=str(checkpoint_dir),
        dataset=str(mango_eval_dataset_dir),
        output_dir=str(tmp_path / "out"),
    )

    with (
        patch.object(inference_runner, "load_model", side_effect=_fake_loaded_model),
        patch.object(
            inference_runner,
            "_generate",
            return_value=inference_runner._GenerationTelemetry(
                text="This mango leaf is affected by anthracnose.",
                elapsed_ms=1.0,
                prompt_tokens=1,
                generation_tokens=1,
                peak_memory_gb=0.0,
                confidence=None,
            ),
        ),
    ):
        out_path = inference_runner.run_inference(cfg)

    rows = [json.loads(line) for line in out_path.read_text().splitlines()]
    assert rows[0]["disease_id"] == "mango_anthracnose"
    assert not rows[0]["disease_id"].startswith("unknown:")
    assert not rows[0]["disease_id"].startswith("tomato_")
    metadata = json.loads((out_path.parent / "metadata.json").read_text())
    assert metadata["crop"] == "mango"


@pytest.mark.unit
def test_run_inference_fails_closed_on_malformed_checkpoint(
    eval_dataset_dir: Path, tmp_path: Path
) -> None:
    from plantdx.core.exceptions import DerivationError
    from plantdx.evaluation import inference_runner

    cfg = resolve_eval_config(
        stage="inference",
        adapter=str(tmp_path / "no_such_checkpoint"),
        dataset=str(eval_dataset_dir),
        output_dir=str(tmp_path / "out"),
    )
    with pytest.raises(DerivationError, match="not found"):
        inference_runner.run_inference(cfg)
