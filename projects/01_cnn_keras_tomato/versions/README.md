# Earlier versions of this project (kept exactly as they were)

The code one level up is the maintained version (paths fixed for this workstation, tested). These folders are the
snapshots behind the report's version table, unchanged, so each step can be compared:

| Version (report) | Date | Folder | What changed |
|---|---|---|---|
| v0.2 – v0.3 | 04 – 06 Jan 2026 | `v0.2-v0.3_cottoncrop_2026-01-06` | CottonCrop repository (`5cc1f6b`): Week-2 `preprocessing/` package, `validation/` crop check, EfficientNetB3 (04 Jan), MobileNetV2 and ConvNeXt-Tiny (06 Jan) |
| v0.5 | 21 – 27 Jan 2026 | `v0.5_integrated_pipeline_2026-01-27` | Kamaaksha's pipeline: the 4–6 Jan models reused byte-identical, reference crop embeddings, `config.py` validate → preprocess → predict with a 0.6 confidence threshold, plus the December custom CNN |

The model files (`*.keras`, `*.npy`) are the same bytes as in `../model/`; `python setup/download_models.py --all`
puts copies here too. The scripts use the paths of the machine they were written on (a Mac for v0.5); to run a
model, use `../predict.py`.
