from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from plantdx.company.external_evaluate import (
    discover_external_images,
    run_external_benchmark,
)
from plantdx.company.predict import Prediction

from .conftest import make_image


def test_discovers_classes_by_name_and_ignores_result_folders(
    tmp_path: Path, company_dataset
) -> None:
    root = tmp_path / "testing"
    make_image(root / "Late_Blight" / "late.jpg", 1)
    make_image(root / "Aphid" / "aphid.jpg", 2)
    make_image(root / "Healthy" / "leaf.jpg", 3)
    make_image(root / "correct" / "old-result.jpg", 4)

    rows, mapping, ignored = discover_external_images(root, company_dataset)

    assert [row.class_id for row in rows] == ["aphid", "late_blight", "healthy_leaf"]
    assert mapping == {
        "Aphid": "aphid",
        "Healthy": "healthy_leaf",
        "Late_Blight": "late_blight",
    }
    assert ignored == ["correct"]


def test_limit_is_balanced_and_positive(tmp_path: Path, company_dataset) -> None:
    root = tmp_path / "testing"
    for class_index, folder in enumerate(("Aphid", "Late_Blight", "Healthy")):
        for image_index in range(3):
            make_image(root / folder / f"{image_index}.jpg", 100 * class_index + image_index)

    rows, _, _ = discover_external_images(root, company_dataset, limit=4)

    assert [row.class_id for row in rows[:3]] == ["aphid", "late_blight", "healthy_leaf"]
    assert len(rows) == 4
    with pytest.raises(Exception, match="limit must be positive"):
        discover_external_images(root, company_dataset, limit=0)


def test_external_benchmark_is_inference_only_and_writes_reports(
    tmp_path: Path, company_config, company_dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    import torch

    import plantdx.company.external_evaluate as external

    root = tmp_path / "testing"
    make_image(root / "Aphid" / "aphid.jpg", 10)
    make_image(root / "Late_Blight" / "late.jpg", 11)
    make_image(root / "Healthy" / "healthy.jpg", 12)
    make_image(root / "wrong" / "prior.jpg", 13)
    report_dir = tmp_path / "report"

    class FakeModel:
        training = True
        gradients_disabled = False

        def requires_grad_(self, enabled: bool):
            self.gradients_disabled = not enabled
            return self

        def eval(self):
            self.training = False
            return self

    model = FakeModel()
    loaded = SimpleNamespace(
        model=model,
        processor=object(),
        model_id="Qwen/Qwen2.5-VL-7B-Instruct",
        adapter_path=str(tmp_path / "adapter"),
    )

    def fake_load(*_args, **_kwargs):
        return loaded

    def fake_predict(_loaded, image_path: Path, **_kwargs):
        assert torch.is_inference_mode_enabled()
        assert not torch.is_grad_enabled()
        assert not model.training
        labels = {
            "Aphid": ("aphid", "Aphid"),
            "Late_Blight": ("aphid", "Aphid"),
            "Healthy": ("healthy_leaf", "Healthy Leaf"),
        }
        class_id, label = labels[image_path.parent.name]
        return Prediction(
            class_id=class_id,
            label=label,
            generated=label,
            confidence=0.9,
            seconds=0.25,
            adapter=loaded.adapter_path,
            base_model=loaded.model_id,
        )

    monkeypatch.setattr(external, "load_predictor", fake_load)
    monkeypatch.setattr(external, "predict_image", fake_predict)

    result = run_external_benchmark(
        company_config,
        company_dataset,
        root,
        output_dir=report_dir,
    )

    assert model.gradients_disabled
    assert result.total_images == 3
    assert result.number_of_classes == 3
    assert result.accuracy == pytest.approx(2 / 3)
    assert result.average_latency_seconds == pytest.approx(0.25)
    assert result.ignored_folders == ("wrong",)
    for filename in (
        "metrics.json",
        "classification_report.csv",
        "confusion_matrix.png",
        "misclassified_examples.csv",
        "predictions.csv",
    ):
        assert (report_dir / filename).is_file()

    metrics = json.loads((report_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["folder_mapping"]["Healthy"] == "healthy_leaf"
    assert metrics["ignored_folders"] == ["wrong"]

    with (report_dir / "classification_report.csv").open(encoding="utf-8", newline="") as handle:
        class_rows = list(csv.DictReader(handle))
    assert list(class_rows[0]) == ["class", "support", "precision", "recall", "f1-score"]
    assert {row["class"] for row in class_rows} == {"Aphid", "Late Blight", "Healthy Leaf"}

    with (report_dir / "predictions.csv").open(encoding="utf-8", newline="") as handle:
        predictions = list(csv.DictReader(handle))
    assert len(predictions) == 3
    assert set(predictions[0]) >= {
        "image",
        "ground_truth_label",
        "predicted_label",
        "confidence",
        "latency_seconds",
    }

    with (report_dir / "misclassified_examples.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        errors = list(csv.DictReader(handle))
    assert errors == [
        {
            "image": str((root / "Late_Blight" / "late.jpg").resolve()),
            "true_label": "Late Blight",
            "predicted_label": "Aphid",
        }
    ]
