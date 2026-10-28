"""Hallucination detection: deterministic, DKB/vocabulary-grounded, reproducible.

Detection is lexicon-based (never an LLM judge -- consistent with the project's
"no LLM/VLM in the pipeline" invariant, and fully reproducible). Four of the six
checks read directly from the frozen DKB / compiled vocabulary artifacts (agent
names, forbidden non-observable symptoms, other-disease names); the crop and
treatment lexicons are small, hand-authored keyword lists, since the training
corpus never emits that language at all -- any occurrence is inherently
suspect, regardless of source.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from plantdx.core.exceptions import DerivationError
from plantdx.evaluation.classification import DiseaseLexicon, build_lexicon

_DEFAULT_DKB = Path("knowledge_base/dkb.json")
_DEFAULT_VOCAB = Path("artifacts/vocabulary/vocabulary.json")

# The training corpus never emits management/treatment language (`management`
# is an always-forbidden concept, see concepts/policies.py); any occurrence is
# a hallucination relative to what the frozen corpus could have taught the model.
_TREATMENT_TERMS = (
    "fungicide",
    "pesticide",
    "insecticide",
    "spray",
    "apply copper",
    "copper spray",
    "treatment",
    "treat with",
    "remove infected",
    "prune",
    "fertilize",
    "fertilizer",
    "quarantine",
    "resistant variety",
    "crop rotation",
)


@dataclass(frozen=True)
class HallucinationFlags:
    """Which hallucination categories fired for one prediction."""

    other_disease: bool
    hallucinated_pathogen: bool
    hallucinated_treatment: bool
    hallucinated_crop: bool
    impossible_symptom: bool

    @property
    def any(self) -> bool:
        """Whether any hallucination category fired for this prediction."""
        return (
            self.other_disease
            or self.hallucinated_pathogen
            or self.hallucinated_treatment
            or self.hallucinated_crop
            or self.impossible_symptom
        )


@dataclass(frozen=True)
class HallucinationLexicons:
    """All grounded lexicons needed to score one crop's predictions."""

    disease_lexicon: DiseaseLexicon
    agent_by_disease: dict[str, tuple[str, ...]]
    forbidden_symptoms_by_disease: dict[str, tuple[str, ...]]
    other_crops: frozenset[str]


def build_hallucination_lexicons(
    crop: str,
    *,
    dkb_path: str | Path = _DEFAULT_DKB,
    vocabulary_path: str | Path = _DEFAULT_VOCAB,
) -> HallucinationLexicons:
    """Build every grounded lexicon needed for hallucination scoring."""
    disease_lexicon = build_lexicon(crop, dkb_path=dkb_path)
    agent_by_disease = _agent_lexicon(vocabulary_path, disease_lexicon.disease_ids)
    forbidden = _forbidden_symptom_lexicon(dkb_path, crop)
    return HallucinationLexicons(
        disease_lexicon=disease_lexicon,
        agent_by_disease=agent_by_disease,
        forbidden_symptoms_by_disease=forbidden,
        other_crops=_other_crops(dkb_path, crop),
    )


def _other_crops(dkb_path: str | Path, crop: str) -> frozenset[str]:
    """Every crop name in the DKB except ``crop`` itself.

    Derived from the DKB rather than a hardcoded list, so a prediction that
    mentions a crop it shouldn't (e.g. "mango" in a tomato evaluation) is
    always flagged, and a legitimate mention of the crop being evaluated is
    never flagged -- for any crop the DKB is later extended with, not just
    tomato and mango.
    """
    data = json.loads(Path(dkb_path).read_text(encoding="utf-8"))
    return frozenset(str(entry["crop"]) for entry in data["diseases"] if entry["crop"] != crop)


def score_hallucinations(
    prediction: str, ground_truth_disease_id: str, lexicons: HallucinationLexicons
) -> HallucinationFlags:
    """Score one prediction's text against the true disease's grounded facts."""
    lowered = prediction.lower()

    other_disease = _mentions_other_disease(lowered, ground_truth_disease_id, lexicons)
    pathogen = _mentions_wrong_pathogen(lowered, ground_truth_disease_id, lexicons)
    treatment = any(term in lowered for term in _TREATMENT_TERMS)
    crop = any(name in lowered for name in lexicons.other_crops)
    impossible = _mentions_forbidden_symptom(lowered, ground_truth_disease_id, lexicons)

    return HallucinationFlags(
        other_disease=other_disease,
        hallucinated_pathogen=pathogen,
        hallucinated_treatment=treatment,
        hallucinated_crop=crop,
        impossible_symptom=impossible,
    )


def _mentions_other_disease(lowered_text: str, truth_id: str, lex: HallucinationLexicons) -> bool:
    for phrase, disease_id in lex.disease_lexicon.all_phrases_longest_first():
        if disease_id != truth_id and phrase in lowered_text:
            return True
    return False


def _mentions_wrong_pathogen(lowered_text: str, truth_id: str, lex: HallucinationLexicons) -> bool:
    own_agents = {a.lower() for a in lex.agent_by_disease.get(truth_id, ())}
    for disease_id, agents in lex.agent_by_disease.items():
        if disease_id == truth_id:
            continue
        for agent in agents:
            agent_l = agent.lower()
            if agent_l and agent_l not in own_agents and agent_l in lowered_text:
                return True
    return False


def _mentions_forbidden_symptom(
    lowered_text: str, truth_id: str, lex: HallucinationLexicons
) -> bool:
    for phrase in lex.forbidden_symptoms_by_disease.get(truth_id, ()):
        if phrase.lower() in lowered_text:
            return True
    return False


def _agent_lexicon(
    vocabulary_path: str | Path, disease_ids: tuple[str, ...]
) -> dict[str, tuple[str, ...]]:
    path = Path(vocabulary_path)
    if not path.is_file():
        raise DerivationError(
            f"compiled vocabulary not found: {path}. Run `plantdx vocabulary` first "
            f"(a frozen, upstream stage; this module only reads its artifact)."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    by_disease: dict[str, list[str]] = {d: [] for d in disease_ids}
    for item in data["items"]:
        if item["concept"] != "agent_name":
            continue
        for ref in item.get("dkb_reference", []):
            if ref in by_disease:
                by_disease[ref].append(str(item["surface_form"]))
    return {k: tuple(v) for k, v in by_disease.items()}


def _forbidden_symptom_lexicon(dkb_path: str | Path, crop: str) -> dict[str, tuple[str, ...]]:
    path = Path(dkb_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, tuple[str, ...]] = {}
    for entry in data["diseases"]:
        if entry["crop"] != crop:
            continue
        raw = entry.get("forbidden_symptoms_not_leaf_observable", [])
        phrases = tuple(sorted({_core_phrase(str(p)) for p in raw if p}))
        result[entry["id"]] = phrases
    return result


def _core_phrase(phrase: str) -> str:
    """Strip a trailing parenthetical clarification from a DKB phrase.

    E.g. 'fruit rot (firm greasy brown fruit lesions)' -> 'fruit rot' -- the
    same parenthetical-leakage fix already applied in the corpus generator (RC1).
    """
    return re.sub(r"\s*\([^)]*\)\s*$", "", phrase).strip()
