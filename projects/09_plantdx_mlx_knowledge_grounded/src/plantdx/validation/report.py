"""Validation context and run-level report models (doc 03 §5)."""

from __future__ import annotations

from dataclasses import dataclass, field

from plantdx.ontology.models import DiseaseOntology
from plantdx.vocabulary.models import DiseaseVocabulary, SymptomLexicon


@dataclass(frozen=True, slots=True)
class ValidationContext:
    """Everything a validator needs besides the caption itself (doc 03 §2)."""

    ontology: DiseaseOntology
    vocabulary: DiseaseVocabulary
    symptom_lexicon: SymptomLexicon
    stage_terms: frozenset[str]
    function_words: frozenset[str]
    scaffold_lexicon: frozenset[str]
    severity_conditioned: bool
    rival_hallmark_terms: dict[str, tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class ValidatorRejection:
    """A single rejected attempt, for the aggregate report."""

    validator_id: str
    attempt: int
    detail: str


@dataclass(frozen=True, slots=True)
class RunReport:
    """Aggregate validation stats for a generation run (doc 03 §5)."""

    library_version: str
    per_disease_rejections: dict[str, dict[str, int]] = field(default_factory=dict)
    per_disease_mean_attempts: dict[str, float] = field(default_factory=dict)
    per_disease_fallback_rate: dict[str, float] = field(default_factory=dict)
    hard_errors: tuple[str, ...] = ()

    @property
    def passes_run_gates(self) -> bool:
        """Placeholder for the run-gate check (fallback ≤ 2%, 0 hard errors)."""
        raise NotImplementedError("Milestone 3: run-gate evaluation")
