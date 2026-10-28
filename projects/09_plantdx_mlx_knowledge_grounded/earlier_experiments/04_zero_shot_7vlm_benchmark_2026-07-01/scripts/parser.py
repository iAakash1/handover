"""
Extracts the labeled fields from a model's raw structured-text response.

Two separate concerns, kept deliberately distinct:

1. Where does a field's captured text END? Answered generically: ANY
   line that looks like "Some Words:" (optionally with a parenthetical
   qualifier, e.g. "(0-100)") is treated as a boundary, whether or not
   it's one of our recognized labels. This matters because if a model
   uses a label we don't recognize at all, that line still needs to be
   treated as *a* boundary — otherwise its text would silently glue onto
   whichever recognized field happened to precede it (e.g. corrupting a
   correctly-parsed Disease field with a neighboring field's stray text).

2. Which fields get POPULATED in the output? Answered via a curated
   whitelist of accepted label variants per field (FIELD_VARIANTS below)
   — e.g. "Confidence", "Confidence Score", "Confidence Level",
   "Certainty" all resolve to the same `confidence` field. Matching is
   case-insensitive and ignores a trailing parenthetical qualifier (so
   "Confidence (0-100)", "Confidence (%)", and "confidence" all match the
   same way). This is still an EXACT match against a known list, not
   fuzzy/regex-guessed content — a label that isn't on the list for any
   field is left unrecognized, and that field stays None rather than
   being guessed at. The complete text is always preserved separately as
   raw_output by the caller, so nothing is ever lost, and a model's
   tendency to deviate from the requested label is itself useful,
   honest signal for the evaluation phase, not something this parser
   should paper over by inventing a value.

Adding another accepted variant for a field, or an entirely new field,
is a one-line change to FIELD_VARIANTS — no other code needs to change.

Known limitation: if free-form Reasoning text itself contains a short
line ending in a colon (e.g. a model-written sub-heading), that line
will be (correctly) treated as a boundary and will end the Reasoning
capture early. This is rare in practice, and raw_output is unaffected —
only the convenience `reasoning` field would be shortened.
"""

import re
from typing import Dict, List, Optional

# Canonical field -> accepted label variants. The first entry in each
# list is the exact label requested in the prompt; the rest are common
# rewordings models are known to use instead.
FIELD_VARIANTS: Dict[str, List[str]] = {
    "crop": ["Crop", "Plant", "Crop Type", "Plant Type"],
    "visible_symptoms": ["Visible Symptoms", "Symptoms", "Observed Symptoms", "Visual Symptoms", "Symptom"],
    "disease": ["Disease", "Diagnosis", "Predicted Disease", "Disease Name", "Condition"],
    "alternative_diagnoses": [
        "Alternative Diagnoses",
        "Alternative Diagnosis",
        "Alternative Diseases",
        "Differential Diagnosis",
        "Differential Diagnoses",
        "Other Possible Diagnoses",
        "Other Possible Diseases",
    ],
    "severity": ["Severity", "Severity Level", "Disease Severity"],
    "confidence": ["Confidence (0-100)", "Confidence", "Confidence Score", "Confidence Level", "Certainty"],
    "reasoning": ["Reasoning", "Explanation", "Justification", "Rationale", "Analysis"],
}

# Broad boundary pattern — matches ANY short "Capitalized phrase:" (any
# case) at the start of a line, with an optional trailing parenthetical
# like "(0-100)" or "(%)", not just our recognized labels. See docstring.
_BOUNDARY_RE = re.compile(
    r"(?:^|\n)[ \t]*([A-Za-z][A-Za-z ]{0,40}?(?:\s?\([^)\n]{0,12}\))?)[ \t]*:",
    re.MULTILINE,
)

# Strips a trailing parenthetical qualifier (e.g. "Confidence (0-100)"
# and "Confidence (%)" both normalize to "confidence"). The boundary
# match above still keeps the parenthetical as part of the consumed
# text, so it can't be mistaken for a second field — this stripping only
# affects how a label is looked up against FIELD_VARIANTS.
_TRAILING_PAREN_RE = re.compile(r"\s*\([^)]*\)\s*$")


def _normalize_label(label_text: str) -> str:
    """'Confidence (0-100)' -> 'confidence'; 'Alternative  Diagnosis' -> 'alternative diagnosis'."""
    stripped = _TRAILING_PAREN_RE.sub("", label_text)
    return " ".join(stripped.split()).lower()


# variant text -> canonical field key, built with the SAME normalization
# used at query time (see parse_response) so both sides of the
# comparison are guaranteed consistent rather than relying on the
# variant list already being pre-normalized by hand.
_VARIANT_TO_KEY: Dict[str, str] = {
    _normalize_label(variant): key for key, variants in FIELD_VARIANTS.items() for variant in variants
}


def parse_response(raw_text: Optional[str]) -> Dict[str, Optional[str]]:
    """
    Parse a raw VLM response into a dict keyed by canonical field name:
    crop, visible_symptoms, disease, alternative_diagnoses, severity,
    confidence, reasoning. A field is populated only when the model used
    one of the accepted label variants for it (FIELD_VARIANTS); anything
    else is left as None (see module docstring for why).

    Never raises on malformed input — worst case, every field comes back
    None and the caller still has the untouched raw_output to fall back on.
    """
    result: Dict[str, Optional[str]] = {key: None for key in FIELD_VARIANTS}

    if not raw_text:
        return result

    matches = list(_BOUNDARY_RE.finditer(raw_text))
    for i, match in enumerate(matches):
        label_text = match.group(1).strip()
        key = _VARIANT_TO_KEY.get(_normalize_label(label_text))
        if key is None:
            continue  # unrecognized label — a boundary, but populates nothing
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)
        value = raw_text[start:end].strip()
        result[key] = value if value else None

    return result
