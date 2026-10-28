"""Clinical correctness: severity honesty, forbidden terminology, agent match."""

from __future__ import annotations

import pytest

from plantdx.evaluation.clinical import score_clinical_correctness


@pytest.mark.unit
@pytest.mark.requires_dkb
class TestClinicalCorrectness:
    def test_severity_claim_is_flagged(self, clinical_lex, hallucination_lex) -> None:
        text = "This leaf shows severe early blight."
        flags = score_clinical_correctness(
            text, "tomato_early_blight", clinical_lex, hallucination_lex
        )
        assert flags.correct_severity_wording is False

    def test_no_severity_claim_passes(self, clinical_lex, hallucination_lex) -> None:
        text = "This leaf shows early blight lesions."
        flags = score_clinical_correctness(
            text, "tomato_early_blight", clinical_lex, hallucination_lex
        )
        assert flags.correct_severity_wording is True

    def test_forbidden_terminology_is_flagged(self, clinical_lex, hallucination_lex) -> None:
        text = "This leaf has water-soaked greasy lesions typical of early blight."
        flags = score_clinical_correctness(
            text, "tomato_early_blight", clinical_lex, hallucination_lex
        )
        assert flags.incorrect_terminology is True

    def test_correct_disease_name_recognized(self, clinical_lex, hallucination_lex) -> None:
        text = "This tomato leaf shows bacterial spot."
        flags = score_clinical_correctness(
            text, "tomato_bacterial_spot", clinical_lex, hallucination_lex
        )
        assert flags.correct_disease is True

    def test_correct_agent_recognized(self, clinical_lex, hallucination_lex) -> None:
        text = "This leaf shows early blight caused by Alternaria solani."
        flags = score_clinical_correctness(
            text, "tomato_early_blight", clinical_lex, hallucination_lex
        )
        assert flags.correct_agent is True
