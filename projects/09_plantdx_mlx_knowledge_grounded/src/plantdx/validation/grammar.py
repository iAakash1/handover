"""Grammar checker adapter for validator V11 (doc 03 §2 V11).

Wraps a pinned grammar backend (LanguageTool by default) behind a small
interface so V11 depends on a stable contract, not a specific library version.
The ``none`` backend falls back to structural checks only.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GrammarIssue:
    """A single grammar finding."""

    category: str
    message: str
    offset: int
    length: int


class GrammarChecker:
    """Adapter over the configured grammar backend.

    Args:
        backend: ``language_tool`` | ``none``.
        language: Locale, e.g. ``en-US``.
        blocking_categories: Categories that cause V11 to fail.
    """

    def __init__(
        self,
        backend: str = "language_tool",
        language: str = "en-US",
        blocking_categories: tuple[str, ...] = (),
    ) -> None:
        """Initialize the checker with the backend, language, and blocking categories."""
        self.backend = backend
        self.language = language
        self.blocking_categories = blocking_categories

    def check(self, text: str) -> tuple[GrammarIssue, ...]:
        """Return grammar issues found in ``text``."""
        raise NotImplementedError("Milestone 3: grammar backend integration")
