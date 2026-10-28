"""Synthetic fixtures for evaluation-pipeline tests.

Everything here is either pure Python or built from the real, frozen,
checked-in DKB (never invented data) -- no mlx-vlm, no real inference. The
compiled vocabulary / symptom lexicon are NOT read from `artifacts/vocabulary/`
(gitignored, generated, absent on a clean checkout / CI) -- they are compiled
fresh, in-process, from the committed DKB using the exact same deterministic
`plantdx vocabulary` pipeline, into a session-scoped temp directory. This is
the same pattern `tests/unit/corpus/conftest.py`'s `bundle` fixture already
uses for the corpus-generation tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_DKB_PATH = _ROOT / "knowledge_base" / "dkb.json"

TOMATO_DISEASES = (
    "tomato_bacterial_spot",
    "tomato_early_blight",
    "tomato_healthy",
    "tomato_late_blight",
)


@pytest.fixture
def lexicon():
    from plantdx.evaluation.classification import build_lexicon

    return build_lexicon("tomato", dkb_path=_DKB_PATH)


@pytest.fixture(scope="session")
def compiled_vocabulary_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Compile vocabulary.json + symptom_lexicon.json fresh from the real DKB.

    Runs the actual compiler pipeline (DKB -> ontology -> vocabulary), the same
    one `plantdx vocabulary` runs, in-process and once per test session, and
    writes its real output artifacts to a temp directory -- never a hardcoded
    path, never a fixture depending on something a clean checkout doesn't have.
    """
    from plantdx.ontology.domain import compile_ontology, validate_ontology
    from plantdx.vocabulary.domain import (
        build_vocabulary_result,
        compute_statistics,
        validate_vocabulary_result,
        write_artifacts,
    )

    onto = compile_ontology(_DKB_PATH)
    validate_ontology(onto)
    vocab = build_vocabulary_result(onto.ontology)
    report = validate_vocabulary_result(vocab, onto.ontology)
    stats = compute_statistics(vocab, report["status"])

    out_dir = tmp_path_factory.mktemp("compiled_vocabulary")
    write_artifacts(vocab, out_dir, stats, report)
    return out_dir


@pytest.fixture
def hallucination_lex(compiled_vocabulary_dir: Path):
    from plantdx.evaluation.hallucination import build_hallucination_lexicons

    return build_hallucination_lexicons(
        "tomato",
        dkb_path=_DKB_PATH,
        vocabulary_path=compiled_vocabulary_dir / "vocabulary.json",
    )


@pytest.fixture
def clinical_lex(compiled_vocabulary_dir: Path):
    from plantdx.evaluation.clinical import build_clinical_lexicons

    return build_clinical_lexicons(
        "tomato",
        dkb_path=_DKB_PATH,
        symptom_lexicon_path=compiled_vocabulary_dir / "symptom_lexicon.json",
    )


def build_synthetic_prediction_rows() -> list[dict]:
    """8 rows across 4 tomato diseases, matching the frozen predictions.jsonl schema.

    A plain function (not a fixture) so both function- and module-scoped
    fixtures/tests can build the same rows without a fixture-scope mismatch.
    """
    ground_truths = {
        "tomato_bacterial_spot": "This tomato leaf shows bacterial spot with small dark lesions.",
        "tomato_early_blight": "This tomato leaf shows early blight with concentric ring lesions.",
        "tomato_healthy": "This is a healthy tomato leaf with no visible symptoms.",
        "tomato_late_blight": "This tomato leaf shows late blight lesions.",
    }
    rows = []
    for i, (disease, gt) in enumerate(ground_truths.items()):
        class_name = disease.replace("tomato_", "")
        for j in range(2):
            rows.append(
                {
                    "image_id": f"{class_name}/img_{j}.JPG",
                    "image_path": f"/data/tomato/processed/{class_name}/img_{j}.JPG",
                    "disease_id": disease,
                    "class_name": class_name,
                    "instruction": "Describe the visible condition of this tomato leaf.",
                    "ground_truth": gt,
                    "base_prediction": "This is a tomato leaf." if j == 0 else gt,
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


@pytest.fixture
def synthetic_prediction_rows() -> list[dict]:
    return build_synthetic_prediction_rows()


@pytest.fixture
def predictions_jsonl(synthetic_prediction_rows: list[dict], tmp_path: Path) -> Path:
    path = tmp_path / "predictions.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for row in synthetic_prediction_rows:
            fh.write(json.dumps(row) + "\n")
    return path


@pytest.fixture(scope="session")
def has_eval_stack() -> bool:
    try:
        import nltk.translate.meteor_score  # noqa: F401
        import pycocoevalcap.bleu.bleu
        import pycocoevalcap.cider.cider  # noqa: F401
        import rouge_score.rouge_scorer  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.fixture(scope="session")
def has_bertscore() -> bool:
    """Whether bert-score can actually load its backbone in this environment.

    Distinct from being pip-installed: a numba/NumPy ABI conflict can still
    block it (see docs/EVALUATION.md#troubleshooting). Session-scoped since the
    backbone-load attempt is expensive and its result is stable for the run.
    """
    try:
        import bert_score

        bert_score.score(["a"], ["a"], lang="en", verbose=False)
    except Exception:
        return False
    return True
