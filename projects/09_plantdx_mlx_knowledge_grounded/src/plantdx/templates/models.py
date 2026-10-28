"""Value objects for the Template Engine.

A template carries *syntax only* — no disease knowledge, no image knowledge. Its
domain content enters solely through slots naming concept ids. Segments make
optional-slot deletion deterministic and grammatical by construction:

* ``lit``  — literal scaffold text.
* ``slot`` — a required concept; the template is unusable unless the concept is
  selected (the Sentence Planner guarantees this).
* ``opt``  — an optional concept; realized as ``glue + phrase`` when selected,
  dropped entirely (glue included) when not.
* ``list`` — an Oxford-comma list of whichever of ``concepts`` were selected.
"""

from __future__ import annotations

from dataclasses import dataclass, field

SEG_LIT = "lit"
SEG_SLOT = "slot"
SEG_OPT = "opt"
SEG_LIST = "list"


@dataclass(frozen=True)
class Segment:
    """One template segment (see module docstring for the four kinds)."""

    kind: str
    text: str = ""  # lit
    concept: str = ""  # slot | opt
    glue: str = ""  # opt | list (prefix emitted before the phrase when present)
    suffix: str = ""  # opt (emitted after the phrase; keeps paired delimiters together)
    conj: str = ""  # list
    concepts: tuple[str, ...] = ()  # list


@dataclass(frozen=True)
class Template:
    """One caption template: a style/register-tagged sequence of segments."""

    id: str
    family: str  # a plantdx.core.enums.Style value
    register: str  # a plantdx.core.enums.Register value
    length_band: str  # a plantdx.core.enums.LengthBand value
    hedged: bool
    sign_type_allow: tuple[str, ...]  # SignType values (or "healthy")
    required: tuple[str, ...]  # concept ids that MUST be present to use this template
    optional: tuple[str, ...]  # concept ids that MAY fill optional/list slots
    segments: tuple[Segment, ...]


@dataclass
class TemplateLibrary:
    """The full authored template library plus provenance."""

    schema_version: str
    template_set_version: str
    families: tuple[str, ...]
    templates: tuple[Template, ...]
    provenance: dict[str, str] = field(default_factory=dict)

    def by_id(self) -> dict[str, Template]:
        """Map ``template_id -> Template`` (ids are unique; enforced by the loader)."""
        return {t.id: t for t in self.templates}
