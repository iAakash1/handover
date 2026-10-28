"""Clinical correctness: disease, symptom, causal-agent, severity, terminology.

Every check is grounded in a frozen artifact (never invented): the DKB's own
`forbidden_adjectives`/`forbidden_terms` fields, the compiled symptom lexicon,
the hallucination module's agent lexicon, and the corpus generator's own
severity-stage token list (`plantdx.concepts.policies.STAGE_TOKENS`) -- the
same list that makes the training corpus itself severity-honest.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from plantdx.core.exceptions import DerivationError
from plantdx.evaluation.hallucination import HallucinationLexicons

_DEFAULT_DKB = Path("knowledge_base/dkb.json")
_DEFAULT_SYMPTOM_LEXICON = Path("artifacts/vocabulary/symptom_lexicon.json")


@dataclass(frozen=True)
class ClinicalLexicons:
    """Grounded facts needed to score one prediction's clinical correctness."""

    symptom_by_disease: dict[str, tuple[str, ...]]
    forbidden_terms_by_disease: dict[str, tuple[str, ...]]
    severity_tokens: tuple[str, ...]


@dataclass(frozen=True)
class ClinicalFlags:
    """Per-sample clinical-correctness outcome."""

    correct_disease: bool
    correct_symptom: bool
    correct_agent: bool
    correct_severity_wording: bool  # True iff no forbidden severity-stage token appears
    incorrect_terminology: bool  # True iff a disease-specific forbidden term appears


def build_clinical_lexicons(
    crop: str,
    *,
    dkb_path: str | Path = _DEFAULT_DKB,
    symptom_lexicon_path: str | Path = _DEFAULT_SYMPTOM_LEXICON,
) -> ClinicalLexicons:
    """Build the grounded lexicons used by :func:`score_clinical_correctness`."""
    from plantdx.concepts.policies import STAGE_TOKENS

    forbidden = _forbidden_terms_lexicon(dkb_path, crop)
    symptoms = _symptom_lexicon(symptom_lexicon_path, set(forbidden.keys()))
    return ClinicalLexicons(
        symptom_by_disease=symptoms,
        forbidden_terms_by_disease=forbidden,
        severity_tokens=tuple(STAGE_TOKENS),
    )


def score_clinical_correctness(
    prediction: str,
    ground_truth_disease_id: str,
    clinical: ClinicalLexicons,
    hallucination_lex: HallucinationLexicons,
) -> ClinicalFlags:
    """Score one prediction against its ground-truth disease's clinical facts."""
    lowered = prediction.lower()

    correct_disease = any(
        phrase in lowered
        for phrase, disease_id in hallucination_lex.disease_lexicon.all_phrases_longest_first()
        if disease_id == ground_truth_disease_id
    )
    own_symptoms = clinical.symptom_by_disease.get(ground_truth_disease_id, ())
    correct_symptom = any(s.lower() in lowered for s in own_symptoms)
    own_agents = hallucination_lex.agent_by_disease.get(ground_truth_disease_id, ())
    correct_agent = any(a.lower() in lowered for a in own_agents)
    no_severity_claim = not any(tok.lower() in lowered for tok in clinical.severity_tokens)
    forbidden = clinical.forbidden_terms_by_disease.get(ground_truth_disease_id, ())
    incorrect_terminology = any(term.lower() in lowered for term in forbidden)

    return ClinicalFlags(
        correct_disease=correct_disease,
        correct_symptom=correct_symptom,
        correct_agent=correct_agent,
        correct_severity_wording=no_severity_claim,
        incorrect_terminology=incorrect_terminology,
    )


def _forbidden_terms_lexicon(dkb_path: str | Path, crop: str) -> dict[str, tuple[str, ...]]:
    path = Path(dkb_path)
    if not path.is_file():
        raise DerivationError(f"DKB not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, tuple[str, ...]] = {}
    for entry in data["diseases"]:
        if entry["crop"] != crop:
            continue
        terms = {str(t).strip() for t in entry.get("forbidden_terms", []) if t}
        terms.update(str(t).strip() for t in entry.get("forbidden_adjectives", []) if t)
        result[entry["id"]] = tuple(sorted(terms))
    return result


def _symptom_lexicon(
    symptom_lexicon_path: str | Path, disease_ids: set[str]
) -> dict[str, tuple[str, ...]]:
    path = Path(symptom_lexicon_path)
    if not path.is_file():
        raise DerivationError(
            f"compiled symptom lexicon not found: {path}. Run `plantdx vocabulary` "
            f"first (a frozen, upstream stage; this module only reads its artifact)."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    by_disease: dict[str, list[str]] = {d: [] for d in disease_ids}
    for item in data["items"]:
        if item["concept"] != "symptom_realization":
            continue
        for ref in item.get("dkb_reference", []):
            if ref in by_disease:
                by_disease[ref].append(str(item["surface_form"]))
    return {k: tuple(v) for k, v in by_disease.items()}
