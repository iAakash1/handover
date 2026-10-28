"""End-to-end stage-2 analysis: every required output file, real content."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.requires_dkb, pytest.mark.requires_eval_stack]

_REQUIRED_FILES = (
    "evaluation_summary.md",
    "evaluation.json",
    "metrics.json",
    "classification_report.csv",
    "per_disease.csv",
    "hallucinations.csv",
    "predictions.csv",
    "confusion_matrix_base.csv",
    "confusion_matrix_finetuned.csv",
    "bleu_scores.csv",
    "rouge_scores.csv",
    "meteor_scores.csv",
    "cider_scores.csv",
    "bertscore.csv",
    "latency.csv",
    "system_info.json",
    "sample_comparisons.md",
    "statistical_comparisons.json",
)
_REQUIRED_FIGURES = (
    "confusion_matrix_base.png",
    "confusion_matrix_base.svg",
    "confusion_matrix_finetuned.png",
    "confusion_matrix_finetuned.svg",
    "accuracy_comparison.png",
    "accuracy_comparison.svg",
    "bounded_metric_comparison.png",
    "cider_comparison.png",
    "bertscore_comparison.png",
    "per_disease_f1.png",
    "hallucination_comparison.png",
    "response_length.png",
    "latency_comparison.png",
)


@pytest.fixture(scope="module")
def analysis_output(tmp_path_factory: pytest.TempPathFactory, has_bertscore: bool):
    if not has_bertscore:
        pytest.skip("bert-score backbone unavailable in this environment (make install-eval)")
    from tests.unit.evaluation.conftest import build_synthetic_prediction_rows

    from plantdx.evaluation.report import run_analysis

    tmp_path = tmp_path_factory.mktemp("eval_report")
    predictions_path = tmp_path / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as fh:
        for row in build_synthetic_prediction_rows():
            fh.write(json.dumps(row) + "\n")

    out_dir = tmp_path / "report"
    written = run_analysis(
        predictions_path,
        output_dir=out_dir,
        model_path="mlx-community/Qwen2.5-VL-7B-Instruct-4bit",
        adapter_path="checkpoints/qwen25vl_tomato_qlora/adapters.safetensors",
        dataset_dir="artifacts/training/qwen25vl_tomato_qlora/dataset",
        seed=20260711,
    )
    return out_dir, written


@pytest.mark.parametrize("filename", _REQUIRED_FILES)
def test_every_required_file_is_written(analysis_output, filename: str) -> None:
    out_dir, _ = analysis_output
    assert (out_dir / filename).is_file(), f"missing {filename}"


@pytest.mark.parametrize("filename", _REQUIRED_FIGURES)
def test_every_required_figure_is_written(analysis_output, filename: str) -> None:
    out_dir, _ = analysis_output
    path = out_dir / "figures" / filename
    assert path.is_file(), f"missing figures/{filename}"
    assert path.stat().st_size > 500


def test_evaluation_json_is_valid_and_complete(analysis_output) -> None:
    out_dir, _ = analysis_output
    data = json.loads((out_dir / "evaluation.json").read_text())
    assert data["sample_count"] == 8
    assert "base" in data["metrics"]
    assert "finetuned" in data["metrics"]
    assert data["reproducibility"]["corpus_checksum"] is not None
    assert len(data["statistical_comparisons"]) > 0


def test_predictions_csv_has_one_row_per_sample(analysis_output) -> None:
    out_dir, _ = analysis_output
    with (out_dir / "predictions.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 8


def test_finetuned_beats_base_on_synthetic_fixture(analysis_output) -> None:
    """The fixture's fine-tuned predictions are verbatim ground truth for every
    sample; base predictions are wrong half the time -- fine-tuned classification
    accuracy must be strictly higher."""
    out_dir, _ = analysis_output
    data = json.loads((out_dir / "metrics.json").read_text())
    base_acc = data["base"]["classification"]["accuracy"]
    ft_acc = data["finetuned"]["classification"]["accuracy"]
    assert ft_acc > base_acc


def test_no_nan_in_statistical_comparisons(analysis_output) -> None:
    out_dir, _ = analysis_output
    comparisons = json.loads((out_dir / "statistical_comparisons.json").read_text())
    text = json.dumps(comparisons)
    assert "NaN" not in text


def test_missing_predictions_file_is_actionable(tmp_path: Path) -> None:
    from plantdx.core.exceptions import PlantDxError
    from plantdx.evaluation.report import run_analysis

    with pytest.raises(PlantDxError, match="not found"):
        run_analysis(
            tmp_path / "nope.jsonl",
            output_dir=tmp_path / "out",
            model_path="m",
            adapter_path="a",
            dataset_dir="d",
            seed=1,
        )


def _mango_prediction_rows() -> list[dict]:
    ground_truths = {
        "mango_anthracnose": "This mango leaf is affected by anthracnose.",
        "mango_healthy": "This is a healthy mango leaf with no visible symptoms.",
    }
    rows = []
    for i, (disease, gt) in enumerate(ground_truths.items()):
        class_name = disease.replace("mango_", "")
        for j in range(2):
            rows.append(
                {
                    "image_id": f"{class_name}/img_{j}.jpg",
                    "image_path": f"/data/mango/processed/{class_name}/img_{j}.jpg",
                    "disease_id": disease,
                    "class_name": class_name,
                    "instruction": "Describe what is shown on this mango leaf image.",
                    "ground_truth": gt,
                    "base_prediction": "This is a mango leaf." if j == 0 else gt,
                    "finetuned_prediction": gt,
                    "base_runtime_ms": 1000.0 + i,
                    "finetuned_runtime_ms": 900.0 + i,
                    "base_prompt_tokens": 40,
                    "base_generation_tokens": 20,
                    "finetuned_prompt_tokens": 40,
                    "finetuned_generation_tokens": 22,
                    "base_peak_memory_gb": 8.5,
                    "finetuned_peak_memory_gb": 8.6,
                    "base_confidence": 0.7,
                    "finetuned_confidence": 0.9,
                }
            )
    return rows


@pytest.fixture(scope="module")
def mango_analysis_output(tmp_path_factory: pytest.TempPathFactory, has_bertscore: bool):
    """Regression fixture for the reported bug: a mango run's `metadata.json`
    (written by stage 1) must drive which crop's lexicon/ontology stage 2
    uses -- never a hardcoded tomato default."""
    if not has_bertscore:
        pytest.skip("bert-score backbone unavailable in this environment (make install-eval)")
    from plantdx.evaluation.report import run_analysis

    tmp_path = tmp_path_factory.mktemp("eval_report_mango")
    predictions_path = tmp_path / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as fh:
        for row in _mango_prediction_rows():
            fh.write(json.dumps(row) + "\n")
    (tmp_path / "metadata.json").write_text(json.dumps({"crop": "mango"}), encoding="utf-8")

    out_dir = tmp_path / "report"
    written = run_analysis(
        predictions_path,
        output_dir=out_dir,
        model_path="mlx-community/Qwen2.5-VL-7B-Instruct-4bit",
        adapter_path="checkpoints/qwen25vl_mango_qlora",
        dataset_dir="artifacts/training/qwen25vl_mango_qlora/dataset",
        seed=20260711,
    )
    return out_dir, written


def test_mango_per_disease_csv_has_mango_ids_only(mango_analysis_output) -> None:
    out_dir, _ = mango_analysis_output
    with (out_dir / "per_disease.csv").open() as fh:
        disease_ids = {row["disease_id"] for row in csv.DictReader(fh)}
    assert "mango_anthracnose" in disease_ids
    assert "mango_healthy" in disease_ids
    assert not any(d.startswith("tomato_") for d in disease_ids)


def test_mango_evaluation_summary_title_says_mango(mango_analysis_output) -> None:
    out_dir, _ = mango_analysis_output
    text = (out_dir / "evaluation_summary.md").read_text()
    assert "(Qwen2.5-VL, mango)" in text
    assert "tomato" not in text.lower()


def test_mango_classification_actually_classifies(mango_analysis_output) -> None:
    """The exact reported symptom: extraction must not collapse to
    "unclassified" for every mango sample just because the lexicon was built
    for the wrong crop."""
    out_dir, _ = mango_analysis_output
    with (out_dir / "predictions.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert all(r["finetuned_predicted_disease"] != "unclassified" for r in rows)
    assert {r["finetuned_predicted_disease"] for r in rows} == {
        "mango_anthracnose",
        "mango_healthy",
    }
