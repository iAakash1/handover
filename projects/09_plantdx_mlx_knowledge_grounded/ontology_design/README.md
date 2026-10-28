# PlantDx — Ontology Layer: Engineering Specification

**Status:** design specification (Milestone "Ontology Design"). No code, no artifacts, no captions.
**Audience:** the engineering team that will implement the Ontology Builder in a later milestone, and reviewers of the IEEE paper.
**Scope:** the *domain ontology* — a deterministic, evidence-linked, crop-independent knowledge graph derived from the Disease Knowledge Base (DKB) and a set of global policies.

This directory is the complete, self-contained design of the ontology layer. It is written so that a second research team could implement it without asking a single design question.

---

## 0. Reading order

| # | Document | What it answers |
|---|----------|-----------------|
| 1 | [01_architecture.md](01_architecture.md) | Why the ontology exists; its place between the DKB and the Caption Framework. |
| 2 | [02_schema.md](02_schema.md) | The `ontology.json` schema: concept types, properties, relations, constraints, cardinality, inheritance, extensibility. |
| 3 | [03_concept_hierarchy.md](03_concept_hierarchy.md) | The complete `is_a` taxonomy of concept types, with closed vs open classes. |
| 4 | [04_concept_graph.md](04_concept_graph.md) | The relation catalog and a fully worked instance subgraph (the semantic graph). |
| 5 | [05_rules.md](05_rules.md) | Allowed/forbidden relations, cardinality, consistency, inheritance, conflict resolution. |
| 6 | [06_statistics.md](06_statistics.md) | Expected node/edge/relation counts, coverage metrics, growth model. |
| 7 | [07_crop_independence.md](07_crop_independence.md) | Why adding a crop never touches the schema. |
| 8 | [08_versioning.md](08_versioning.md) | `schema_version`, `ontology_version`, compatibility, migration. |
| 9 | [09_validation.md](09_validation.md) | What makes an ontology valid/invalid; the automated validator battery. |
| 10 | [10_build_algorithm.md](10_build_algorithm.md) | Deterministic DKB → ontology construction (pseudocode only). |
| 11 | [11_cli_contract.md](11_cli_contract.md) | `plantdx ontology` arguments, artifacts, exit codes, logging. |
| 12 | [12_testing_strategy.md](12_testing_strategy.md) | Unit, integration, regression, property-based tests. |
| 13 | [13_paper_contribution.md](13_paper_contribution.md) | The methodological contribution vs templates / LLM / VLM captions. |

## 1. Terminology (used consistently across all documents)

- **T-Box (schema / terminology box):** the crop-independent *type system* — the concept types, relation types, and constraints. Hand-designed once; frozen except by explicit `schema_version` change.
- **A-Box (assertion box / instances):** the *graph* — the nodes and edges instantiated deterministically from the DKB. Grows as crops/diseases are added; never hand-edited.
- **Node:** a typed instance (an individual). Every node has an `id`, a `type` (a T-Box concept type), and `properties`.
- **Edge:** a typed, directed relation between two nodes, with `attributes` (confidence, evidence references, licensing flags).
- **Closed class:** a concept type whose individuals are a fixed enumeration owned by a global policy (e.g., `Severity = {mild, moderate, severe}`). Never populated from the DKB.
- **Open class:** a concept type populated from the DKB (e.g., `Disease`, `Symptom`, `Color` values).
- **Value node (canonical individual):** a shared node for a controlled-vocabulary value (e.g., `color:brown`) referenced by many symptoms — the mechanism that makes this a graph, not a forest.
- **Assertion licensing:** a per-edge policy flag stating whether an assertion is licensed *by the dataset label alone* (`asserted`), is *characteristic of the class but not guaranteed in a specific image* (`typical`), or is *rare/secondary* (`hedged`). This is the epistemic backbone of the paper.

## 2. Non-negotiable design principles

1. **Determinism.** `Ontology = f(DKB, GlobalPolicies)`, a pure function. The same inputs always produce byte-identical output (content-hashed). No randomness.
2. **No learned components.** Nothing depends on an LLM or a VLM. Every node and edge traces to a DKB field or a global policy.
3. **Schema/instance separation.** The type system (T-Box) is crop-independent and hand-designed; the graph (A-Box) is DKB-derived. Adding a crop adds instances, never types.
4. **Language-free representation.** The ontology represents knowledge as typed nodes/edges and controlled-vocabulary value nodes — never as free text. Natural-language rendering is a *downstream* concern (captions), not an ontology concern.
5. **Evidence at the edge.** Every domain assertion carries a reference to an `Evidence` node resolving to the DKB `reference_registry`. Unsupported assertions are invalid.
6. **Observability is structural.** Leaf-observability (single-leaf image constraint) is encoded as graph structure (`appears_on` a leaf vs non-leaf part), not as a comment. Downstream stages *query* it.
7. **Severity honesty is structural.** Per-image severity claims are gated by an edge licensing flag; the ontology never asserts a per-image stage.

## 3. Relationship to prior milestones (one paragraph)

The **DKB** (`knowledge_base/dkb.json`) is flat, per-disease, human-readable records — the *single source of truth for facts*. This **domain ontology** re-expresses those facts as a *formal, queryable, evidence-linked graph* with a fixed type system. The **Caption Framework** (`caption_framework/`, frozen) defines how captions are generated; its per-disease "caption concept model" is henceforth understood as a **deterministic view (projection) over this ontology** — the ontology is the substrate, the caption concept model is one query result. This milestone does not modify the DKB or the Caption Framework; it inserts the formal knowledge-representation layer they both implicitly assumed.
