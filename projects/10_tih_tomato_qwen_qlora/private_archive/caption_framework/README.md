# PlantDx — Caption Generation Framework (Engineering Specification)

**Stage 2 of PlantDx.** This directory is the complete, self-contained engineering specification for the knowledge-grounded caption generation pipeline. It is a **design/methodology deliverable**: no code, no captions, no datasets are produced here. An implementing engineer (Claude Sonnet Max) should be able to build the entire system from these documents **without making additional design decisions**.

The upstream **Disease Knowledge Base (DKB)** at [`../knowledge_base/dkb.json`](../knowledge_base/dkb.json) and [`../knowledge_base/DKB_report.md`](../knowledge_base/DKB_report.md) is **FINAL and the single source of truth**. This framework never restates disease facts; it *derives* everything it needs from the DKB. If the DKB and any document here appear to disagree, the DKB wins and the derivation rule here is the bug.

## Documents (read in this order)

| # | File | Covers | Task(s) |
|---|------|--------|---------|
| 1 | [`00_methodology_overview.md`](00_methodology_overview.md) | System architecture, generation algorithm, information budget, **severity-honesty policy**, reproducibility, **diversity strategy** | Overview, Task 5 |
| 2 | [`01_caption_ontology_spec.md`](01_caption_ontology_spec.md) | Concept schema, **DKB→ontology derivation**, per-disease ontology, **vocabulary expansion lattice** | Task 1, Task 3 |
| 3 | [`02_template_spec.md`](02_template_spec.md) | Slot grammar, 52 templates across 8 styles, selection logic | Task 2 |
| 4 | [`03_validation_spec.md`](03_validation_spec.md) | 12-stage validator battery, regeneration loop, symptom lexicon | Task 4 |
| 5 | [`04_dataset_schema_spec.md`](04_dataset_schema_spec.md) | Canonical caption record, instruction design, **per-model format converters** | Task 6 |
| 6 | [`05_quality_assurance_protocol.md`](05_quality_assurance_protocol.md) | Sampling plan, reviewer checklist, hallucination taxonomy, acceptance sampling | Task 8 |
| 7 | [`06_folder_structure_spec.md`](06_folder_structure_spec.md) | Full runtime directory layout, purpose of every file, data flow | Task 7 |
| 8 | [`07_ieee_methodology_section.md`](07_ieee_methodology_section.md) | Publication-ready methodology section + justification vs LLM/VLM captioning | Task 9 |

## The seven design invariants (non-negotiable)

Every component in this framework exists to preserve these. They are repeated in each document's header as the acceptance contract.

1. **Label-only grounding.** A caption is a function of `(disease_label, DKB)` — never of image pixels, never of a VLM/LLM prediction. The PlantVillage / MangoLeafBD folder label is the sole ground truth.
2. **DKB is the single source of truth.** All allowed concepts, vocabulary, and forbidden terms are *derived from* `dkb.json`. No disease fact is hand-authored in Stage 2.
3. **Closed vocabulary.** In the symptom/description register, only terms present in the disease's DKB vocabulary fields (or the global synonym classes filtered by that disease's `forbidden_terms`) may appear. Open-ended generation is prohibited.
4. **Observability constraint.** No caption may assert anything listed in the disease's `forbidden_symptoms_not_leaf_observable` (fruit, twig, flower, whole-tree, yield, vascular, insect-adult, etc.). Captions describe only single-leaf-observable signs.
5. **Pest/pathogen register integrity.** For classes with `is_pathogen_disease: false` (spider mites, cutting weevil, gall midge, sooty mould), pest/mechanical/surface language is used; the words *infection, pathogen, lesion* (where DKB forbids them) never appear.
6. **Severity honesty.** Because severity is not known per image, captions do **not** assert a per-image severity stage (mild/moderate/severe) unless an explicit per-image severity label is supplied. Extent descriptors that are directly visible (e.g., *scattered, numerous*) are treated separately from stage claims. See §5 of `00_methodology_overview.md`.
7. **Reproducibility.** Generation is fully deterministic given a global seed. Every caption stores complete provenance (template id, selected concepts, vocabulary choices, seed) so it can be regenerated bit-for-bit and audited.

## What "no VLM" buys us (one line)

The circularity that dooms VLM-generated captions — using a model that is bad at the task to create the labels that will teach that task — is structurally impossible here, because no model output ever enters the pipeline. Full argument in `07_ieee_methodology_section.md`.
