# April 2026: Qwen2.5-VL, the model that worked

## Work done

- Read research papers (four downloaded on 6 Apr).
- Florence-2 fine-tuning attempted (3 Apr); stopped by library incompatibilities.
- Qwen2.5-VL-7B-Instruct fine-tuned with LoRA r32 (6–7 Apr).
- Production pipeline (13 Apr): CLIP check that rejects non-leaf photos (threshold 0.27), output parser, plant-mismatch validator, GradCAM heat-map.

## Findings

- Qwen2.5-VL LoRA: **98.50%** accuracy, macro-F1 0.9827, BLEU-4 0.4556, severity R² 0.9654.
- The same model without fine-tuning: 8.50%, so fine-tuning provides almost all of the accuracy.
- Florence-2 zero-shot: 0.00%; it describes scenes, not diseases.

## Results

| Model / evaluation set | Accuracy | Macro-F1 |
|---|---:|---:|
| Qwen2.5-VL-7B LoRA r32 (200) | 98.50% | 0.9827 |
| Qwen2.5-VL-7B zero-shot (200) | 8.50% | 0.0800 |
| Florence-2-base-ft zero-shot (200) | 0.00% | 0.0000 |

## Evaluation reports in this folder

| File | Copied from |
|---|---|
| [`2026-03_March__qwen__evaluation_report.json`](evaluation_reports/2026-03_March__qwen__evaluation_report.json) | `2026-04_April/qwen/evaluation_report.json` |
| [`qwen_zeroshot__evaluation_report.json`](evaluation_reports/qwen_zeroshot__evaluation_report.json) | `evaluation_reports_oct2026/qwen_zeroshot/evaluation_report.json` |
| [`florence_zeroshot__evaluation_report.json`](evaluation_reports/florence_zeroshot__evaluation_report.json) | `evaluation_reports_oct2026/florence_zeroshot/evaluation_report.json` |

## Code and models

- [Qwen2.5-VL code and adapter](../2026-04_April/qwen)
- [production pipeline](../2026-04_April/latest_changes)
- [Florence-2 attempt](../2026-04_April/florence)
