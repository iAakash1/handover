# Vocabulary + Symptom Lexicon Compiler

A CPU-only, **deterministic** compiler: `Vocabulary = f(Ontology, Policies)`. It
reads the compiled domain ontology (never the DKB directly) plus the fixed
category/relation mappings in `src/plantdx/vocabulary/domain/policies.py`, and
projects it into two flat, traceable artifacts: a controlled **vocabulary**
(colors, shapes, textures, extents, leaf regions, sign types, agent names,
disease names, environment descriptors, confidence and observability
modifiers) and a bounded **symptom lexicon** (realized symptom phrases such as
"brown lesion"). It uses no images, no datasets, no LLM/VLM, and no
randomness — repeated builds are byte-identical.

> This is the **domain vocabulary compiler** (`plantdx.vocabulary.domain`). It
> is separate from the caption-concept vocabulary stub in `plantdx.vocabulary`
> (an unimplemented M1 scaffold) — the two coexist in the `plantdx.vocabulary`
> package, mirroring how `plantdx.ontology.domain` coexists with `plantdx.ontology`.

## Design

- **Vocabulary is a projection, not a generator.** Every vocabulary item is one
  ontology node, read through the one relation type that grounds it (see the
  `CATEGORIES` table in `policies.py`). Nothing is invented, expanded, or
  guessed — no synonym generation, no WordNet, no heuristics.
- **The symptom lexicon is bounded, never combinatorial.** Each symptom gets
  its own verbatim DKB phrase (the *base* realization) plus, only for
  *primary* symptoms whose sign type is modifiable (`lesion`, `coating`,
  `gall`, `stippling` — Caption Framework §2.4's co-selection rule), one
  realization per quality value already attached to the condition
  (`<modifier> <head noun>`, e.g. "brown lesion"). This is linear in the
  number of attached qualities, never a Cartesian product across color × shape
  × texture × extent. Multi-modifier stacking ("circular brown lesion") is
  deferred to a future Vocabulary Expander.
- **Cross-axis word collisions are deduplicated deterministically.** A few DKB
  conditions reuse the same word across two quality axes (e.g.
  `shape:raised` and `texture:raised` on the same condition) — legitimate,
  distinct ontology facts that would otherwise realize as the same phrase
  twice for one symptom. When that happens, only the highest-priority axis
  survives, in `MODIFIER_RELATIONS` order (color, shape, texture, extent); the
  dropped axis is still present, verbatim, as its own vocabulary item.
- **Every item is traceable.** Each entry carries the exact metadata schema
  `id, surface_form, canonical_form, concept, concept_id, confidence, source,
  ontology_node, dkb_reference, evidence, language, part_of_speech` — so any
  word or phrase can be traced back through its ontology node and grounding
  relation to the DKB disease(s) and citations that licensed it.

## Run it

From the repository root (`experiments/`):

```bash
plantdx vocabulary                       # compile + validate + write artifacts
plantdx vocabulary --validate-only       # compile + validate; write nothing
plantdx vocabulary --stats-only          # compile + validate; print statistics JSON
plantdx vocabulary --output some/dir     # override output directory
python -m plantdx vocabulary             # equivalent, without the console script
```

## Artifacts

Written to `artifacts/vocabulary/` (gitignored):

| File | Contents |
|------|----------|
| `vocabulary.json` | The controlled vocabulary: one item per covered ontology node, plus the 3 fixed confidence modifiers. |
| `symptom_lexicon.json` | The bounded symptom lexicon: one base + N single-modifier realizations per symptom. |
| `concept_index.json` | Lookup indices over both artifacts: by concept, by ontology node, by DKB disease reference. |
| `statistics.json` | Counts by concept/confidence, coverage, validation status, checksum. |
| `checksum.txt` | The content-only SHA-256 (`sha256:…`); identifies the build. |
| `validation_report.json` | The validator battery's pass report (status, checks run, item counts). |

## Exit codes

| Code | Meaning |
|------|---------|
| `0` | Success (valid vocabulary + lexicon built / validated). |
| `1` | Validation failure — either the source ontology or the vocabulary itself violated a rule (fail closed; no artifacts written). |
| `2` | Configuration or DKB load/validation error (from compiling the source ontology). |

## Validation battery

Fail closed: any violation aborts the build before artifacts are written.

| Rule | Guards against |
|------|-----------------|
| `V-VOC-1` | Duplicate item ids. |
| `V-VOC-2` | Duplicate realizations (two items, one symptom, same surface text). |
| `V-VOC-3` | Orphan concepts (`concept_id`/`ontology_node` not resolvable in the ontology). |
| `V-VOC-4` | Evidence presence that disagrees with the category's evidence-carrying contract. |
| `V-VOC-5` | Illegal combinations (a modifier realization using a non-modifier relation, or modifying a non-modifiable sign type). |
| `V-VOC-6` | Illegal modifiers (a modifier attached to a non-primary symptom). |
| `V-VOC-7` | Invalid realizations (schema violations: empty/whitespace-damaged text, bad confidence value, etc.). |
| `V-VOC-8` | Unused concepts (an ontology node in a covered category with no vocabulary item). |
| `V-VOC-9` | Conflicting realizations (an item's label disagrees with an independent re-derivation from its ontology node). |

## Guarantees

- **Deterministic.** No timestamps in artifacts, no UUIDs, everything sorted; the
  checksum depends only on vocabulary + lexicon content (not location/machine/OS/time).
- **Ontology-only input.** The builder never reads the DKB, an image, or a model —
  only the compiled `Ontology` returned by `plantdx.ontology.domain`.
- **Evidence-linked.** Every evidence-carrying category's items resolve, through
  their grounding relation, back to `Evidence` nodes in the DKB `reference_registry`.
- **Fail closed.** Any rule violation aborts the build with a specific `V-VOC-*` error.
