# August 2026: expert-caption training and the TIH dataset

## Work done

- PlantVLM: Qwen2.5-VL trained only on the 150 expert captions (30 images, 6 diseases) (5 Aug).
- TIH private tomato dataset received (12 Aug); PlantDx migrated to it.
- SigLIP / OpenCLIP prototype pipeline written for the TIH dataset (27 Aug).

## Findings

- PlantVLM named the right disease for 2 of 6 validation images (chance 1 of 6): 30 images are too few.
- The SigLIP/OpenCLIP prototype was not trained; Qwen2.5-VL QLoRA was chosen for the TIH data.

## Results

| Model / evaluation set | Accuracy | Macro-F1 |
|---|---:|---:|
| PlantVLM retrieval (6) | 33.33% | 0.2778 |

## Evaluation reports in this folder

| File | Copied from |
|---|---|
| [`run_002_fold0__evaluation__val_report.json`](evaluation_reports/run_002_fold0__evaluation__val_report.json) | `2026-08_August/newpipe/runs/run_002_fold0/evaluation/val_report.json` |
| [`run_002_fold0__logs__epochs.jsonl`](evaluation_reports/run_002_fold0__logs__epochs.jsonl) | `2026-08_August/newpipe/runs/run_002_fold0/logs/epochs.jsonl` |

## Code and models

- [PlantVLM](../2026-08_August/newpipe)
- [SigLIP/OpenCLIP prototype](../2026-08_August/siglip_openclip_tih_prototype)
