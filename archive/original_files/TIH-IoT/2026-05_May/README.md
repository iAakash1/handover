# May 2026: PlantDx

## Work done

- Restructured the April pipeline into PlantDx: 15 PlantVillage classes (tomato, potato, pepper), modular code, Streamlit app (15–18 May).

## Findings

- 98.00% accuracy, macro-F1 0.9777, BLEU-4 0.386, severity R² 0.966: the same quality as April with reusable code.

## Results

| Model / evaluation set | Accuracy | Macro-F1 |
|---|---:|---:|
| PlantDx Qwen2.5-VL LoRA (200) | 98.00% | 0.9777 |

## Evaluation reports in this folder

| File | Copied from |
|---|---|
| [`outputs__evaluations__report.json`](evaluation_reports/outputs__evaluations__report.json) | `2026-05_May/PlantDx/outputs/evaluations/report.json` |

## Code and models

- [PlantDx code and adapter](../2026-05_May/PlantDx)
