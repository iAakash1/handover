"""Caption Generator (component F->realization).

Consumes a :class:`SentencePlan` and produces the final English caption. It only
concatenates already-chosen controlled phrases and applies deterministic surface
repair (whitespace, punctuation, articles, capitalization, Oxford-comma lists,
terminal punctuation). It never invents, infers, or adds domain content — every
word is either template scaffolding or a phrase the Sentence Planner selected
from the concept model. Deterministic: the same plan always yields the same text.
"""

from __future__ import annotations

import re

from plantdx.corpus.models import PIECE_CONCEPT, PIECE_LIST, PIECE_LIT, PlanPiece, SentencePlan

# Abbreviations whose trailing period is not a sentence boundary (protect species
# names like "Capnodium spp." and "X. citri pv. mangiferaeindicae").
_ABBREVIATIONS = frozenset({"spp", "sp", "pv", "var", "subsp", "cf", "al", "eg", "ie", "no"})
_VOWELS = frozenset("aeiouAEIOU")


def generate(plan: SentencePlan) -> str:
    """Realize a sentence plan into a normalized English caption."""
    raw = "".join(_render_piece(p) for p in plan.pieces)
    return _normalize(raw)


def _render_piece(p: PlanPiece) -> str:
    if p.kind == PIECE_LIT:
        return p.text
    if p.kind == PIECE_CONCEPT:
        return f"{p.glue}{p.phrase}{p.suffix}"
    if p.kind == PIECE_LIST:
        return f"{p.glue}{_oxford([phrase for _, phrase in p.items], p.conj)}"
    return ""


def _oxford(phrases: list[str], conj: str) -> str:
    """Join phrases as an Oxford-comma list."""
    if not phrases:
        return ""
    if len(phrases) == 1:
        return phrases[0]
    if len(phrases) == 2:
        return f"{phrases[0]} {conj} {phrases[1]}"
    return ", ".join(phrases[:-1]) + f", {conj} {phrases[-1]}"


def _normalize(text: str) -> str:
    """Deterministic surface repair so any slot-deletion still reads grammatically."""
    s = re.sub(r"\s+", " ", text).strip()
    s = _collapse_repeats(s)  # "raised raised" / "on the lamina on the lamina" -> once
    s = re.sub(r"\(\s+", "(", s)  # no space after "("
    s = re.sub(r"\s+([,.;:!?)])", r"\1", s)  # no space before closing punctuation
    s = re.sub(r"[,;:]\s*(?=[.;:])", "", s)  # drop a comma/semicolon before another stop
    s = re.sub(r"([,;:])\1+", r"\1", s)  # collapse repeated separators
    s = re.sub(r"\.\.+", ".", s)  # collapse ellipses introduced by joins
    s = re.sub(r"\ban\b(?=\s+[^aeiouAEIOU\s])", "a", s)  # "an leaf" -> "a leaf"
    s = _fix_articles(s)
    s = s.strip()
    if s and s[-1] not in ".!?":
        s += "."
    return _capitalize_sentences(s)


def _collapse_repeats(s: str) -> str:
    """Collapse an immediately repeated word or short span (a slot-join artifact)."""
    for span in (3, 2, 1):  # longest span first so "on the lamina on the lamina" wins
        inner = r"\w+" + r"(?:\s+\w+)" * (span - 1)
        s = re.sub(rf"\b({inner})\s+\1\b", r"\1", s, flags=re.IGNORECASE)
    return s


def _fix_articles(s: str) -> str:
    """Deterministic a/an agreement before a vowel-initial word."""

    def repl(m: re.Match[str]) -> str:
        article, nxt = m.group(1), m.group(2)
        target = "an" if article.islower() else "An"
        return f"{target} {nxt}" if nxt[0] in _VOWELS else m.group(0)

    return re.sub(r"\b([Aa]) ([A-Za-z]\w*)", repl, s)


def _capitalize_sentences(s: str) -> str:
    """Capitalize the first letter and each true sentence start (protecting abbrevs)."""
    chars = list(s)
    n = len(chars)
    for i in range(n):  # first alphabetic character
        if chars[i].isalpha():
            chars[i] = chars[i].upper()
            break
    for i in range(n - 1):
        if chars[i] in ".!?" and chars[i + 1] == " ":
            j = i + 2
            while j < n and chars[j] == " ":
                j += 1
            if j < n and chars[j].islower() and not _is_abbreviation_period(chars, i):
                chars[j] = chars[j].upper()
    return "".join(chars)


def _is_abbreviation_period(chars: list[str], i: int) -> bool:
    """Whether the period at index ``i`` closes an abbreviation, not a sentence."""
    if i == 0:
        return False
    # Single capital letter (genus abbreviation like "C.").
    if chars[i - 1].isupper() and (i - 1 == 0 or not chars[i - 2].isalpha()):
        return True
    # A known lowercase abbreviation ("spp.", "pv.", ...).
    j = i - 1
    word: list[str] = []
    while j >= 0 and chars[j].isalpha():
        word.append(chars[j])
        j -= 1
    return "".join(reversed(word)).lower() in _ABBREVIATIONS
