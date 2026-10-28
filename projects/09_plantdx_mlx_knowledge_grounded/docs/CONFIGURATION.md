# Configuration Reference

All configuration lives in [`../configs/`](../configs/) and is loaded/validated by
`plantdx.config.load_config` into the typed `PlantDxConfig`
(`src/plantdx/config/schema.py`).

## Precedence

```
code defaults  <  YAML (configs/)  <  PLANTDX_* environment variables  <  CLI flags
```

## Files

| File | Owns | Spec |
|------|------|------|
| `config.yaml` | project identity, `includes:`, logging, reproducibility | doc 00 §9 |
| `paths.yaml` | logical→physical path mapping (the single mapping layer) | doc 06 |
| `generation.yaml` | seeds, caption budget, style/task distributions, anti-domination, diversity gates, splits | doc 00 §7, doc 06 §4 |
| `validation.yaml` | the 12 validators, matching rules, grammar backend, run gates | doc 03 |
| `training.yaml` | QLoRA/MLX hyperparameters per model, evaluation | doc 04 §6 |

## Key tunables (defaults)

| Key | Default | Meaning |
|-----|---------|---------|
| `generation.global_seed` | `20260711` | Root of all determinism (doc 00 §6). |
| `generation.captions_per_image` | `3` | `K` captions per image. |
| `generation.max_attempts` | `8` | Regeneration budget before minimal fallback. |
| `generation.hedging_probability` | `0.9` | Probability a secondary sign is hedged (keep high). |
| `generation.severity_conditioned` | `false` | **Never** true without per-image severity labels (invariant #6). |
| `dedup.jaccard_threshold` | `0.90` | Near-duplicate rejection threshold. |
| `diversity_gates.*` | see file | Hard acceptance gates (doc 00 §7.7). |
| `splits.{train,val,test}` | `0.80/0.10/0.10` | Image-grouped, stratified split ratios. |
| `validation.grammar.backend` | `language_tool` | V11 backend (`none` = structural only). |

## Environment overrides

Copy `.env.example` to `.env` (gitignored). Recognized: `PLANTDX_ARTIFACT_ROOT`,
`PLANTDX_GLOBAL_SEED`, `PLANTDX_LOG_LEVEL`, `PLANTDX_DKB_PATH`.

## Golden rule

**No magic numbers in code.** Every tunable is in `configs/`. Changing a config
value bumps `library_version` and requires re-validation/QA before release.
