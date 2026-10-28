# March 2026: VLM papers and first VLM fine-tuning

## Work done

- Studied Vision-Language Model papers for crop disease (list in the main README).
- Fine-tuned LLaVA-1.6-Mistral-7B and BLIP-2 OPT-2.7B with LoRA on PlantVillage structured captions (24–25 Mar).

## Findings

- BLIP-2 reached 89.00% accuracy (macro-F1 0.8928) on the shared 200-image test.
- LLaVA-1.6 reached only 2.00%: it wrote long generic descriptions and named no disease in 178 of 200 images.
- Conclusion: a model that follows the output format is needed, which led to Qwen2.5-VL in April.

## Results

| Model / evaluation set | Accuracy | Macro-F1 |
|---|---:|---:|
| BLIP-2 OPT-2.7B LoRA (200) | 89.00% | 0.8928 |
| LLaVA-1.6-Mistral-7B LoRA (200) | 2.00% | 0.0151 |

## Evaluation reports in this folder

| File | Copied from |
|---|---|
| [`blip2_march__evaluation_report.json`](evaluation_reports/blip2_march__evaluation_report.json) | `evaluation_reports_oct2026/blip2_march/evaluation_report.json` |
| [`llava_v16__evaluation_report.json`](evaluation_reports/llava_v16__evaluation_report.json) | `evaluation_reports_oct2026/llava_v16/evaluation_report.json` |

## Code and models

- [LLaVA and BLIP-2 adapters](../2026-03_March/outputs)
