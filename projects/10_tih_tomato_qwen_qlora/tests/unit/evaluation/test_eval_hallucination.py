"""DKB/vocabulary-grounded hallucination detection."""

from __future__ import annotations

from pathlib import Path

import pytest

from plantdx.evaluation.hallucination import build_hallucination_lexicons, score_hallucinations


@pytest.fixture
def mango_hallucination_lex(compiled_vocabulary_dir: Path):
    return build_hallucination_lexicons(
        "mango", vocabulary_path=compiled_vocabulary_dir / "vocabulary.json"
    )


@pytest.mark.unit
@pytest.mark.requires_dkb
class TestHallucinationDetection:
    def test_faithful_prediction_has_no_flags(self, hallucination_lex) -> None:
        text = "This tomato leaf shows bacterial spot with small dark lesions."
        flags = score_hallucinations(text, "tomato_bacterial_spot", hallucination_lex)
        assert flags.any is False

    def test_treatment_language_is_flagged(self, hallucination_lex) -> None:
        text = "This leaf shows early blight, apply fungicide to treat it."
        flags = score_hallucinations(text, "tomato_early_blight", hallucination_lex)
        assert flags.hallucinated_treatment is True

    def test_other_disease_mention_is_flagged(self, hallucination_lex) -> None:
        text = "This leaf shows late blight, possibly also target spot."
        flags = score_hallucinations(text, "tomato_late_blight", hallucination_lex)
        assert flags.other_disease is True

    def test_other_crop_is_flagged(self, hallucination_lex) -> None:
        text = "This mango leaf shows anthracnose lesions."
        flags = score_hallucinations(text, "tomato_bacterial_spot", hallucination_lex)
        assert flags.hallucinated_crop is True

    def test_wrong_pathogen_is_flagged(self, hallucination_lex) -> None:
        text = "This leaf shows early blight caused by Phytophthora infestans."
        flags = score_hallucinations(text, "tomato_early_blight", hallucination_lex)
        assert flags.hallucinated_pathogen is True

    def test_correct_pathogen_is_not_flagged(self, hallucination_lex) -> None:
        text = "This leaf shows early blight caused by Alternaria solani."
        flags = score_hallucinations(text, "tomato_early_blight", hallucination_lex)
        assert flags.hallucinated_pathogen is False

    def test_impossible_symptom_is_flagged(self, hallucination_lex) -> None:
        text = "This tomato leaf shows fruit rot typical of late blight."
        flags = score_hallucinations(text, "tomato_late_blight", hallucination_lex)
        assert flags.impossible_symptom is True

    def test_own_crop_mention_is_not_flagged(self, mango_hallucination_lex) -> None:
        """Regression test: a mango evaluation must never flag a legitimate
        mention of "mango" as a hallucinated crop. The old implementation used
        a single static _OTHER_CROPS list that unconditionally included every
        known crop name including whichever one was actually being evaluated."""
        text = "This mango leaf is affected by anthracnose."
        flags = score_hallucinations(text, "mango_anthracnose", mango_hallucination_lex)
        assert flags.hallucinated_crop is False

    def test_other_crop_is_flagged_for_mango_evaluation_too(self, mango_hallucination_lex) -> None:
        """The old static _OTHER_CROPS list never included "tomato" at all
        (it was only ever used for tomato evaluations), so a mango run would
        never have caught a tomato-crop hallucination."""
        text = "This tomato leaf is affected by anthracnose."
        flags = score_hallucinations(text, "mango_anthracnose", mango_hallucination_lex)
        assert flags.hallucinated_crop is True
