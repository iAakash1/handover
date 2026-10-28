# 1. Ontology Architecture

## 1.1 Why the ontology exists

The DKB answers *"what is true about disease X?"* in prose fields. The Caption Framework answers *"what may a caption of disease X say?"*. Between them lies an unstated dependency: a **formal model of the domain** that (a) makes the DKB machine-queryable without natural-language parsing at consumption time, (b) makes every downstream decision (which symptoms are observable, which are hallmark, what may be said with what confidence) a *graph query* rather than an ad-hoc rule, and (c) makes the whole thing reproducible and citable.

Without this layer, every downstream stage would re-parse DKB prose, re-encode the observability rule, and re-derive confidence — duplicating logic, drifting over time, and defeating traceability. The ontology exists to make the domain knowledge **explicit, typed, deterministic, and shared**.

Concretely, the ontology is the single artifact that lets us state, and mechanically verify, claims the paper depends on:
- *"Every symptom asserted in a caption is licensed by the dataset label and traceable to a peer-reviewed or extension source."* → a reachability query over `Condition –has_symptom→ Symptom –supported_by→ Evidence`.
- *"No caption asserts a feature that is not visible in a single-leaf image."* → a query filtering symptoms whose `appears_on` is a non-leaf part.
- *"The system never claims a per-image severity stage."* → an invariant that no `typical_at_severity` edge is ever surfaced as image-licensed.

## 1.2 What problem it solves

| Problem (without the ontology) | Solution (with the ontology) |
|--------------------------------|------------------------------|
| DKB prose must be re-parsed by every consumer. | Facts are pre-parsed into typed nodes/edges once, deterministically. |
| Observability rule re-implemented per stage. | Observability is a structural property of the graph; queried, not re-coded. |
| Confidence/uncertainty implicit in prose. | Confidence is a first-class edge attribute (`asserted`/`typical`/`hedged`). |
| No provenance from a claim to a source. | Every assertion edge references an `Evidence` node → `reference_registry`. |
| Taxonomic disagreements silently resolved. | Disagreements represented as parallel `caused_by` edges with distinct evidence. |
| Adding a crop risks redesign. | Adding a crop adds A-Box instances; the T-Box is untouched. |
| "Is the caption grounded?" is unverifiable. | Grounding becomes a decidable graph property (see §1.5). |

The ontology is **not** a database convenience; it is the *scientific instrument* that turns "we grounded the captions" from a claim into a checkable property.

## 1.3 Position in the pipeline

```
   knowledge_base/dkb.json            global policies (assets/ontology/*)
   (facts, prose, 46 fields)          (closed vocabularies, observability
            │                          lexicon, severity split, sign-type map)
            └───────────────┬───────────────────────┘
                            ▼
                 ┌─────────────────────┐
                 │   ONTOLOGY BUILDER   │   deterministic f(DKB, policies)
                 │   (later milestone)  │
                 └─────────┬───────────┘
                           ▼
        ┌──────────────────────────────────────────┐
        │  DOMAIN ONTOLOGY  (this specification)    │
        │  T-Box: concept + relation types          │
        │  A-Box: nodes + edges (evidence-linked)   │
        └───────────────┬──────────────────────────┘
                        │  queries / projections (deterministic views)
        ┌───────────────┼───────────────┬───────────────────────┐
        ▼               ▼               ▼                       ▼
  Caption concept   Controlled       Validation           Paper artifacts
  model (a VIEW)    vocabulary view   context view         (graphs, tables,
  per disease       per disease       per disease           coverage metrics)
```

Downstream stages **consume the ontology, not the DKB directly** (the DKB remains the source of truth *for the builder*). This mirrors the normalization milestone's rule that consumers use the normalized datasets, not `raw/`.

## 1.4 Relationship with the DKB

- **Direction:** strictly `DKB → Ontology`. The ontology never writes back to the DKB. The DKB `dkb_sha256` is embedded in the built ontology's provenance; if the DKB changes, the ontology is rebuilt and re-versioned.
- **Field mapping:** every one of the DKB's 46 per-disease fields is either (a) *consumed* into nodes/edges, or (b) *explicitly declared non-structural* (e.g., free-prose `disease_progression`, retained as a node property but not decomposed). The build report lists which fields were consumed and how (see [10_build_algorithm.md](10_build_algorithm.md)); a validator asserts no field is silently dropped (see [09_validation.md](09_validation.md)).
- **Fidelity:** the ontology is a *lossless-enough* re-encoding for downstream purposes: any consumer that previously read a DKB field can obtain the equivalent via a graph query. The ontology may *add* structure (e.g., canonical value nodes) but never *invents* facts.
- **Disagreements:** the DKB's `documented_taxonomic_disagreements` (metadata) and per-disease `taxonomy_note` are represented as multiple `caused_by` edges (one per candidate agent) each carrying its own `Evidence` and `confidence`, rather than being flattened. The ontology preserves scientific honesty.

## 1.5 Relationship with the Caption Framework

The Caption Framework is **frozen**; this milestone does not change it. It is *re-founded*, not redesigned:

- The Caption Framework's per-disease **caption concept model** (its required/optional/forbidden concept sets, its controlled vocabulary, its `never_appear` list) is, from now on, defined as a **deterministic projection of the ontology**. For example:
  - *Required concepts* = the disease's hallmark symptoms and identity nodes reachable with `confidence = asserted`.
  - *Optional concepts* = symptoms with `confidence ∈ {typical, hedged}` that are `observable`.
  - *Forbidden concepts / `never_appear`* = symptoms with `observable = false` (non-leaf `appears_on`), plus rival diseases' hallmark symptom signatures reachable via `differentiated_from`.
  - *Controlled vocabulary* = the canonical value nodes (`Color`, `Shape`, `Texture`, `Distribution`) reachable from the disease's observable symptoms.
  - *Severity vocabulary licensing* = governed by the `image_licensed` flag on `typical_at_severity` / `Extent` edges.
- **Consequence:** the caption framework's outputs are unchanged, but they now have a *formal derivation*. "Why is `fruit lesion` forbidden for early blight?" has a one-line, checkable answer: *because the `fruit lesion` symptom node's `appears_on` is `Fruit`, a non-leaf part, so it is `observable = false`.*

### Why insert this layer now (trade-off analysis)

- **Alternative A — consume the DKB directly (status quo).** Simpler short-term, but every stage re-implements parsing + observability + confidence; the grounding claim stays informal; adding crops is risky. Rejected: it does not scale to "hundreds of diseases, dozens of crops" and gives the paper no verifiable instrument.
- **Alternative B — a full OWL/RDF ontology with a reasoner.** Maximally standard and interoperable, but introduces a heavyweight dependency, non-trivial nondeterminism in reasoner output across versions, and a steep implementation cost inconsistent with the project's "simplest correct implementation" ethos. Rejected as the primary representation; **retained as an optional export target** (§1.6).
- **Chosen — a constrained typed property graph (a "profile").** A minimal, JSON-expressible schema (concept types, relation types, constraints) with a deterministic builder. Gives 90% of the semantic-web benefit (typed nodes, typed relations, domain/range, inheritance, constraints, provenance) at 10% of the cost, and is trivially deterministic. This is the load-bearing decision of the whole design.

## 1.6 Interoperability (deliberately kept optional)

The property-graph model is designed so that a **lossless mapping to RDF/OWL/SKOS** is possible but not required:
- Concept types ↔ `owl:Class`; `is_a` ↔ `rdfs:subClassOf`.
- Relation types ↔ `owl:ObjectProperty` with `rdfs:domain`/`rdfs:range`.
- Closed vocabularies (`Severity`, `LeafRegion`, `Color` …) ↔ `skos:ConceptScheme` with `skos:Concept` members.
- Edge attributes (confidence, evidence) ↔ RDF reification or RDF-star (`<< s p o >>` with qualifiers).

Providing an RDF export is a *future*, additive capability (a serializer over the same A-Box); it never becomes a build dependency. This keeps the core deterministic and dependency-light while leaving the door open to the semantic-web ecosystem for external reuse — important for a "reusable agricultural knowledge representation framework."

## 1.7 What the ontology deliberately does NOT do

- It does **not** store or generate natural language beyond canonical labels and retained DKB prose properties. Sentence realization is the caption stage.
- It does **not** perform inference beyond the deterministic, monotonic constraint checks in [05_rules.md](05_rules.md). There is no probabilistic or learned reasoning.
- It does **not** read pixels, model images, or depend on the normalized datasets' contents (only their *labels* are the ground truth, already fixed upstream).
- It does **not** decide per-image facts (severity, presence of a given symptom in a specific photo). It encodes *class-level* knowledge plus *licensing* that downstream honesty depends on.
