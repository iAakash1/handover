"""App-layer inference logic: confidence, status buckets, adapter verification.

These import ``app.inference`` (which imports Streamlit) but never run a model —
they test the pure logic. Skipped where Streamlit isn't installed (e.g. a
``.[dev]``-only CI environment), mirroring the repo's optional-dependency pattern.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("streamlit", reason="demo app dependency (not in the dev-only env)")

from app.inference import (
    STATUS_CONFIDENT,
    STATUS_LOW_CONFIDENCE,
    STATUS_UNKNOWN,
    _selected_token_prob,
    _status,
    adapter_info,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("disease_id", "confidence", "threshold", "expected"),
    [
        ("tomato_early_blight", 0.9, 0.55, STATUS_CONFIDENT),
        ("tomato_early_blight", 0.4, 0.55, STATUS_LOW_CONFIDENCE),
        ("unclassified", 0.99, 0.55, STATUS_UNKNOWN),  # no disease named -> unknown regardless
        ("tomato_early_blight", None, 0.55, STATUS_CONFIDENT),  # no confidence -> not downgraded
    ],
)
def test_status_buckets(disease_id: str, confidence, threshold: float, expected: str) -> None:
    assert _status(disease_id, confidence, threshold) == expected


def test_selected_token_prob_is_probability_of_emitted_token() -> None:
    logprobs = [math.log(0.5), math.log(0.25), math.log(0.25)]
    chunk = SimpleNamespace(logprobs=logprobs, token=0)
    assert _selected_token_prob(chunk) == pytest.approx(0.5)
    chunk2 = SimpleNamespace(logprobs=logprobs, token=1)
    assert _selected_token_prob(chunk2) == pytest.approx(0.25)


def test_selected_token_prob_handles_missing_fields() -> None:
    assert _selected_token_prob(SimpleNamespace(logprobs=None, token=0)) is None
    assert _selected_token_prob(SimpleNamespace(logprobs=[0.0], token=None)) is None
    assert _selected_token_prob(SimpleNamespace(logprobs=[math.log(0.9)], token=5)) is None  # OOB


def test_adapter_info_reads_config_and_counts_params(tmp_path: Path) -> None:
    save_file = pytest.importorskip("safetensors.numpy").save_file
    np = pytest.importorskip("numpy")

    ckpt = tmp_path / "qwen25vl_tomato_qlora"
    ckpt.mkdir()
    (ckpt / "adapter_config.json").write_text(
        json.dumps(
            {
                "fine_tune_type": "lora",
                "lora_parameters": {"rank": 8, "scale": 2.0, "keys": ["a", "b", "c"]},
            }
        ),
        encoding="utf-8",
    )
    save_file(
        {
            "x.lora_a": np.zeros((4, 8), dtype="float16"),
            "x.lora_b": np.zeros((8, 4), dtype="float16"),
        },
        str(ckpt / "adapters.safetensors"),
    )

    adapter_info.cache_clear()
    info = adapter_info(str(ckpt))
    assert info["attached"] is True
    assert info["fine_tune_type"] == "lora"
    assert info["rank"] == 8
    assert info["num_target_modules"] == 3
    assert info["trainable_params"] == 64  # 4*8 + 8*4
    assert info["lora_tensor_count"] == 2
    assert info["weights_checksum"].startswith("sha256:")


def test_adapter_info_without_weights_reports_not_attached(tmp_path: Path) -> None:
    ckpt = tmp_path / "half"
    ckpt.mkdir()
    (ckpt / "adapter_config.json").write_text('{"fine_tune_type": "lora"}', encoding="utf-8")
    adapter_info.cache_clear()
    info = adapter_info(str(ckpt))
    assert info["attached"] is False
    assert info["trainable_params"] is None
