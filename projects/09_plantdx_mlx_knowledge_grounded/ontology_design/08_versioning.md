# 8. Versioning Strategy

Two independent version numbers, because the *schema* (contract) and the *graph* (data) change for different reasons and are consumed differently.

## 8.1 The two versions

| Version | Governs | Format | Bumped when |
|---------|---------|--------|-------------|
| `schema_version` | The T-Box: concept types, relation types, constraints, closed-vocab *shape*. | **SemVer** `MAJOR.MINOR.PATCH` | Any change to the type system or constraints. |
| `ontology_version` | The built A-Box artifact for a given DKB + policies + schema. | Monotonic tag `O{n}` + embedded `content_hash` | Any change to DKB, policies, schema, or builder that changes the output. |

Both are written into `ontology_graph.json`; `schema_version` is also in `ontology_schema.json`. Consumers **pin `schema_version`** (a contract) and **record `ontology_version`/`content_hash`** (for provenance/reproducibility).

## 8.2 `schema_version` — SemVer semantics

- **PATCH** (`1.0.0 → 1.0.1`): documentation/label fixes, added *optional* metadata that no consumer parses. No structural change.
- **MINOR** (`1.0.0 → 1.1.0`): **additive, backward-compatible** — a new concept type (leaf under an existing parent), a new relation type, a new closed-vocab individual, or a new *optional* property. Existing consumers keep working (they ignore what they don't query). This is the common case for new crops/diseases that need a new sign type or region.
- **MAJOR** (`1.0.0 → 2.0.0`): **breaking** — removing/renaming a concept type, relation, or individual; making a property required; changing a relation's domain/range/cardinality in a narrowing way; changing edge-attribute semantics. Requires a migration ([8.4](#84-migration-policy)).

The builder refuses to emit a graph whose nodes/edges reference a T-Box that a consumer pinned at an incompatible MAJOR (fail fast, [09_validation.md](09_validation.md)).

## 8.3 `ontology_version` and content addressing

- `ontology_version` is a human-friendly incrementing tag (`O1`, `O2`, …) assigned when a build is *published*.
- The *identity* of a build is its **`content_hash`** = deterministic hash over canonically-ordered `(nodes ⧺ edges)` (excludes timestamps). Two builds with the same DKB + policies + schema + builder have the **same** `content_hash` even if built on different days. This is the reproducibility guarantee.
- Provenance embedded in every graph: `dkb_sha256`, `policy_hash`, `schema_hash`, `builder_version`, `content_hash`, `created_utc`. The first four determine the output; `content_hash` verifies it; `created_utc` is informational only (never hashed).

**Relationship to the library-version concept used elsewhere in the repo:** the ontology's `content_hash` plays the same role for the ontology that the caption library's `library_version` plays for captions — a frozen, QA-able identity. Downstream stages that consume the ontology record its `content_hash`.

## 8.4 Backward compatibility policy

- **Additive-by-default.** New knowledge (crops/diseases) must land as MINOR schema changes at most, usually as A-Box-only changes (no schema bump at all).
- **Deprecation, not deletion.** To retire a concept type or relation, first mark it `deprecated: true` (a PATCH/MINOR), keep it emitting for one MAJOR cycle, then remove in the next MAJOR. Consumers get a window.
- **Frozen builds are immutable.** Once an `ontology_version` is QA-accepted and used to build a caption library, it is never edited in place; changes produce a new `ontology_version`.
- **Consumers declare a compatible range**, e.g. `requires_schema: ">=1.2,<2"`. The build (and CI) checks this against `schema_version`.

## 8.5 Migration policy

When a MAJOR schema change is unavoidable:

1. **Write a migration note** `migrations/<from>-<to>.md` describing the structural delta and the rationale (why breaking was necessary).
2. **Provide a mechanical migration mapping** (a declarative table, later implemented): old-type → new-type, old-relation → new-relation, dropped-fields, renamed individuals. Mechanical because the graph is typed and regular.
3. **Rebuild, don't transform.** Because the ontology is a *pure function of the DKB*, the canonical migration is simply: update the T-Box + policies, then **rebuild from the DKB** under the new schema. In-place graph transformation is offered only for external consumers who hold an old graph they cannot rebuild.
4. **Dual-publish** the old and new `ontology_version` for one release so downstream stages can migrate on their own cadence.
5. **Migration tests** ([12_testing_strategy.md](12_testing_strategy.md)) assert that rebuilding an old DKB under the new schema yields a graph that passes all validators and preserves every prior *fact* (no knowledge lost across the migration).

## 8.6 Determinism as a versioning guarantee

Because `Ontology = f(DKB, policies)` is pure and content-hashed:
- A version *cannot* silently drift: any change to inputs changes the `content_hash`, forcing a new `ontology_version`.
- A claimed reproduction is *verifiable*: rebuild and compare `content_hash`.
- The `builder_version` is part of provenance so that a builder bugfix that changes output is itself a visible, versioned event.

This makes the ontology's version history a faithful, auditable record — a requirement for the artifact to underpin a reproducible benchmark.
