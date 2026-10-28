# Pull Request

## Summary
<!-- What does this PR do? Which milestone/component? -->

## Related
- Milestone:
- Spec reference (caption_framework/… or knowledge_base/…):

## Checklist
- [ ] Scoped to a single milestone/component
- [ ] `make check` passes (ruff format + lint, mypy strict, pytest)
- [ ] Public APIs are typed and have Google-style docstrings
- [ ] New behavior covered by tests
- [ ] Preserves the seven design invariants (label-only grounding, DKB single source
      of truth, closed vocabulary, observability, pest/pathogen register, severity
      honesty, reproducibility)
- [ ] No disease facts / vocabulary / forbidden terms hard-coded in `.py` files
- [ ] Deterministic (no unseeded randomness in the caption path)
- [ ] CHANGELOG updated

## Notes for reviewers
<!-- Anything that needs special attention -->
