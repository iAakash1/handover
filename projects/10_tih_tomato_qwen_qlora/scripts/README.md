# scripts/

Thin wrappers around the `plantdx` CLI and common developer flows. They contain
**no logic** — the logic lives in the package (`src/plantdx/`). Prefer the CLI
(`plantdx …`) directly; these exist for convenience and reproducible invocations.

| Script | Purpose |
|--------|---------|
| `dev_setup.sh` | Create a venv, install `.[dev]`, install pre-commit hooks. |
| `run_pipeline.sh` | Run the end-to-end pipeline stages in order (stubbed until each milestone lands). |
| `setup_eval_env.sh` | Install the `.[eval]` metrics stack + cache WordNet / BERTScore once. |
| `render_readme_figures.py` | Re-render the README evaluation figures from the existing `reports/<run>/evaluation/` CSVs (reads real values only; needs matplotlib). |

Make the shell scripts executable with `chmod +x scripts/*.sh`.
