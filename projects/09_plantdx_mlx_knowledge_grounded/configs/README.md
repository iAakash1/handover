# configs/

Authored configuration for PlantDx. Loaded and merged by
[`plantdx.config.load_config`](../src/plantdx/config/loader.py); validated against
the typed schema in [`plantdx.config.schema`](../src/plantdx/config/schema.py).

| File | Purpose | Spec reference |
|------|---------|----------------|
| `config.yaml` | Master config; composes the others; project + logging + reproducibility. | doc 00 §9 |
| `paths.yaml` | The single mapping layer: logical artifact layout → physical repo paths. | doc 06 |
| `generation.yaml` | Seeds, caption budget, style/task distributions, anti-domination, diversity gates, splits. | doc 00 §7, doc 06 §4 |
| `validation.yaml` | The 12-stage validator battery, matching rules, grammar backend, run gates. | doc 03 |
| `training.yaml` | QLoRA/MLX hyperparameters per target model; evaluation settings. | doc 04 §6 |

Precedence (low → high): code defaults < these YAML files < `PLANTDX_*` env vars < CLI flags.

**Do not put magic numbers in code.** Every tunable is here. Changing any of these
bumps `library_version` (see `docs/REPO_LAYOUT.md`) and requires re-validation/QA.
