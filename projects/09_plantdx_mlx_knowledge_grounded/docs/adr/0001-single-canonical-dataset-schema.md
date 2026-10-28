# ADR 0001 — Single canonical dataset schema + per-model adapters

- **Status:** Accepted
- **Date:** 2026-07-11
- **Spec:** `caption_framework/04_dataset_schema_spec.md` §7

## Context

Four target models (Qwen2.5-VL, Qwen3-VL, InternVL3, Gemma-3) plus the MLX
(`mlx-vlm`) toolchain expect different training-file schemas. We must decide
whether to generate separate datasets per model or one canonical dataset with
converters.

## Decision

Maintain **one canonical, tool-agnostic caption record** (`core/types.CaptionRecord`,
serialized by `dataset/serialization.py`) as the single source of truth, and derive each
trainer's files with **pure, deterministic converters** (`dataset/converters/`)
at build time. Converters add only role scaffolding and image placeholders; they
never alter caption or instruction text.

## Consequences

- All four models train on **identical content and identical image-level splits**
  — a precondition for a fair zero-shot-vs-fine-tuned comparison (Stage 6).
- Adding a model = writing one adapter; a trainer-format change (notably
  `mlx-vlm`'s shifting schema) is isolated to a single function.
- Provenance and QA are done once on the canonical library, not N times.
- Trade-off: an extra conversion step at Stage 4 (cheap, pure, independently
  testable). Accepted.
