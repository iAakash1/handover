# June 2026: publication of PlantDx and BLIP-2 re-run

## Work done

- PlantDx code published on GitHub (2–3 Jun).
- BLIP-2 re-run for 1 epoch with full evaluation (15 Jun); Florence-2 re-attempted.
- Expert-validated caption image set prepared (24 Jun); seven-VLM zero-shot benchmark (25 Jun).

## Findings

- The 1-epoch BLIP-2 re-run was scored 0% by its own June script; with the April protocol it reaches 89.00% (macro-F1 0.8948). The March run reached 89.00%.
- Zero-shot benchmark (PoCRA report): Qwen3-VL-8B produced degenerate output on 16 of 30 images; general VLMs are unreliable without fine-tuning.

## Results

| Model / evaluation set | Accuracy | Macro-F1 |
|---|---:|---:|
| BLIP-2 1-epoch re-run, April protocol (200) | 89.00% | 0.8948 |

## Evaluation reports in this folder

| File | Copied from |
|---|---|
| [`blip2_june__evaluation_report.json`](evaluation_reports/blip2_june__evaluation_report.json) | `evaluation_reports_oct2026/blip2_june/evaluation_report.json` |
| [`2026-03_March__tp__evaluation_report.json`](evaluation_reports/2026-03_March__tp__evaluation_report.json) | `2026-06_June/tp/evaluation_report.json` |

## Code and models

- [BLIP-2 re-run](../2026-06_June/tp)
- [PoCRA report (zero-shot benchmark)](../docs/06-PoCRA-PCCOE-Tomato-Mango-Report-Version1.docx)
