# Architecture Decision Records (ADRs)

Short, immutable records of significant decisions. Format: Context → Decision →
Consequences. Add a new numbered file for each decision; never rewrite history —
supersede instead.

| # | Title | Status |
|---|-------|--------|
| [0001](0001-single-canonical-dataset-schema.md) | Single canonical dataset schema + per-model adapters | Accepted |
| [0002](0002-src-layout-and-artifact-mapping.md) | `src/` layout with an artifact mapping layer | Accepted |

> Decisions that change the DKB or caption-framework methodology are out of
> scope: those specifications are FINAL. ADRs here record *implementation*
> decisions consistent with the spec.
