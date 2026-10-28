# 13. Paper Contribution

This document states, precisely, what the ontology contributes methodologically — the claim the IEEE paper makes and defends.

## 13.1 One-sentence contribution

> We introduce a **deterministic, evidence-linked, crop-independent knowledge graph** that serves as the grounding substrate for agricultural vision–language caption generation, turning "the captions are grounded" from an assertion into a **machine-checkable property** and making the entire dataset **reproducible from a single source of truth**.

## 13.2 The methodological gap it fills

Prior approaches to producing image-description supervision for agricultural VLMs fall into three families, each with a structural weakness the ontology removes:

| Approach | How captions arise | Structural weakness |
|----------|-------------------|---------------------|
| **Rule-based templates** | Hand-written templates + slot fills. | No formal knowledge model: symptom/observability/confidence logic is scattered in template code, not represented or checkable; brittle; no provenance; hard to extend to new crops without rewriting rules. |
| **LLM-generated captions** | Prompt an LLM per label/image. | Non-deterministic; hallucinates symptoms not licensed by the label; no provenance from claim to source; unverifiable grounding; drift across model versions. |
| **VLM-generated captions** | Caption the image with a VLM, then train on that. | Circular (student inherits teacher errors); not grounded in ground-truth pathology; the earlier zero-shot benchmark showed general VLMs are unreliable at crop-disease diagnosis, so distilling them encodes their mistakes. |

The ontology is the missing **explicit knowledge layer**: it makes the domain knowledge a first-class, typed, sourced, queryable object that *precedes and constrains* any caption. Captions become a *rendering* of licensed knowledge, not a *guess*.

## 13.3 Contribution vs each baseline (specific)

### vs rule-based templates
- Templates encode *syntax*; the ontology encodes *semantics*. The template's slot values are no longer chosen by ad-hoc rules but *queried* from the graph (observable, confidence-tagged, sourced). The observability rule and severity policy live **once**, structurally, instead of being re-implemented per template.
- Extensibility: a new crop is new DKB entries → new subgraph → templates unchanged. Rule-based systems require new rules per crop.
- Verifiability: "why is this slot value allowed?" has a graph-path answer; templates offer none.

### vs LLM-generated captions
- **Determinism:** `Ontology = f(DKB, policies)` is a pure function with a content hash; an LLM is a distribution over outputs. Reproducibility is definitional here, impossible there.
- **No hallucination by construction:** every asserted symptom is a node with `observable=true`, `confidence=asserted`, and ≥1 `Evidence` edge. There is no path by which an unlicensed claim enters, because the caption stage draws only from the licensed set.
- **Provenance:** each claim → `Evidence` node → citation. LLM captions have no claim-to-source mapping.

### vs VLM-generated captions
- **Non-circular:** the ontology is built from *curated pathology* (DKB + peer-reviewed/extension sources), never from a model's image interpretation. The supervision does not inherit a VLM's errors.
- **Ground-truth-aligned:** captions describe what the *label* licenses (the dataset ground truth), plus what is *visible in a single leaf* (observability), not what a model "sees."

## 13.4 Why this improves reproducibility

- **Pure function + content hash.** Same DKB + policies ⇒ byte-identical ontology (`content_hash`). Anyone can rebuild and verify. There is no hidden state, no sampling, no model weights.
- **Versioned, immutable builds.** `schema_version` + `ontology_version` + provenance (`dkb_sha256`, `policy_hash`) make every build auditable and re-derivable ([08_versioning.md](08_versioning.md)).
- **Deterministic downstream.** Because the caption concept model is a *view* over the ontology, and the caption generator is itself seeded and deterministic, the *entire* dataset is reproducible from `(DKB, policies, schema, seed)` — a property no LLM/VLM pipeline can claim.

## 13.5 Why this improves traceability

- **Evidence at the edge.** Every domain assertion references a source in the DKB `reference_registry` (APS, UC IPM, UF/IFAS, CABI, peer-reviewed papers). A reviewer can trace any caption claim → symptom node → evidence → citation.
- **Coverage is reported, not assumed.** The build emits DKB field-coverage and source-quality metrics ([06_statistics.md](06_statistics.md) §6.3): e.g., "% of hallmark symptoms with peer-reviewed/extension support." These become table rows in the paper.
- **Disagreements are preserved.** Disputed pathogen taxonomy is represented as parallel, separately-sourced `caused_by` edges — the system does not hide scientific uncertainty; it records it.

## 13.6 Why this improves scientific validity

- **Observability is structural.** The single-leaf-image constraint — the crux of honest agricultural captioning — is a graph property (`appears_on` a leaf vs non-leaf part), not a comment. Captions cannot assert fruit/twig/whole-plant symptoms because those symptom nodes are `observable=false` and excluded by query. This is *the* scientific-validity guarantee, and it is checkable (V-ONT-8).
- **Severity honesty is structural.** The system never claims a per-image disease stage, because `typical_at_severity` edges are permanently `image_licensed=false`. Class-level severity knowledge is retained (not discarded) but never asserted about a specific photo.
- **Register integrity is structural.** Pest damage / superficial colonization / disease are distinct `Condition` subtypes with distinct permitted sign types, so a caption cannot call spider-mite stippling a "lesion" or sooty-mould coating a "necrosis." These pathology-correctness rules are enforced at the knowledge layer.

## 13.7 The reusable-framework contribution (beyond this paper)

The ontology is designed as a **reusable agricultural knowledge-representation framework**, not a one-off:
- The **T-Box is crop- and disease-agnostic**; any leaf-disease DKB in the 46-field schema instantiates it. New crops and diseases are additive ([07_crop_independence.md](07_crop_independence.md)).
- It scales to the stated horizon (hundreds of diseases, dozens of crops, millions of captions) with **no architectural change** ([06_statistics.md](06_statistics.md) §6.4), because the ontology is queried per-*class*, not per-*caption*, and its vocabulary saturates.
- It offers an **optional RDF/OWL/SKOS mapping** ([01_architecture.md](01_architecture.md) §1.6) for external interoperability without taking on a heavyweight dependency internally.

The paper's contribution is therefore two-fold: (1) a *method* — deterministic, evidence-grounded knowledge-to-caption pipelines that are reproducible and traceable by construction; and (2) an *artifact* — a released, versioned ontology and builder specification that other agricultural-VLM researchers can adopt and extend.

## 13.8 Threats to validity (stated honestly)

- **DKB fidelity ceiling.** The ontology is only as correct as the DKB; it faithfully re-encodes, it does not fact-check pathology. Mitigation: the DKB itself is sourced and reviewed; the ontology surfaces low-evidence assertions as metrics.
- **Deterministic parsing limits.** Symptom/quality extraction uses deterministic vocabulary matching, not language understanding; a phrase outside the controlled vocabulary is surfaced for human policy extension, never guessed. This trades recall for guaranteed precision/reproducibility — the correct trade-off for a scientific dataset.
- **Abstraction of anatomy.** Modeling functional leaf regions (not full botanical morphology) is a deliberate simplification; it is sufficient for foliar-disease captioning but would need extension for, e.g., whole-plant or fruit datasets — an additive, versioned change, not a redesign.
