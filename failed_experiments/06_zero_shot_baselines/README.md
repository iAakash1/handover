# 06 · VLMs without fine-tuning (zero-shot)

Before and after each fine-tuning run we asked the untouched models the same question. They don't know plant
diseases:

| Model | Data | Accuracy | Report |
|---|---|---:|---|
| Qwen2.5-VL-7B-Instruct | PlantVillage, 200 validation images | 8.50% | `evaluation/qwen_zeroshot/` |
| Qwen2.5-VL-7B-Instruct | TIH tomato, 649 validation images | 8.47% (macro-F1 0.0766; 587 answers named no class) | `../08_tih_aborted_training_run/reports/validation_base/` |
| Florence-2-base-ft | PlantVillage, 200 validation images | 0.00% | `evaluation/florence_zeroshot/` |
| Four VLMs on MLX, 25 Jun 2026 | 16 leaf images (tomato, potato, pepper, cotton) | exact disease match 0–6% per model; right disease mentioned anywhere 19–50% | `mlx_4vlm_benchmark_2026-06-25/report/` |
| Seven VLMs on MLX, 1 Jul 2026 | 30 expert-checked tomato images | best accuracy 23% (InternVL3-8B); Qwen3-VL-8B gave degenerate output on 16 of 30 | `mlx_7vlm_benchmark_2026-07-01/report/`, PoCRA report `docs/reports/06-PoCRA-PCCOE-Tomato-Mango-Report-Version1.docx` |

The same Qwen2.5-VL went to 98.50% (PlantVillage) and 92.76% (TIH) after LoRA/QLoRA fine-tuning.

## The two MLX benchmarks (Apple Silicon, Mac only)

Both ran off-the-shelf 4-bit models with `mlx-vlm` on an Apple M4 Pro (24 GB), one fixed prompt, no fine-tuning.
They need macOS on Apple Silicon (`pip install -r requirements.txt` in a Python 3.12 env); the models download
from Hugging Face (`mlx-community/...`) on first use.

| Folder | Models | Images | What is in it |
|---|---|---|---|
| `mlx_4vlm_benchmark_2026-06-25/` | Qwen2.5-VL-7B, Gemma-3-12B, InternVL3-8B, Llama-3.2-11B-Vision | 16 (`images/`) | `scripts/` (benchmark, ground truth from file names, LLM judge, evaluation, plots, report), raw answers per model (`results/`, `json_results/`), `report/final_report.md` + PDF + charts |
| `mlx_7vlm_benchmark_2026-07-01/` | the four above + Qwen3-VL-8B, Molmo-7B-D, DeepSeek-VL2-small | 30 (`dataset/`, 6 diseases × 5) | `scripts/`, every prediction (`outputs/`, `results/predictions.csv`), per-model logs, `report/PlantDx_VLM_Evaluation_Report.docx` (scored against the 150 expert captions of project 08) |

```bash
cd failed_experiments/06_zero_shot_baselines/mlx_4vlm_benchmark_2026-06-25 && bash run_all.sh
cd failed_experiments/06_zero_shot_baselines/mlx_7vlm_benchmark_2026-07-01 && python scripts/run_benchmark.py
```

The LLM-judge step of the four-model benchmark needs an `OPENAI_API_KEY` or `GEMINI_API_KEY`; none was set, so
the report has no judge scores (it is skipped automatically). Speed on the M4 Pro: Qwen2.5-VL-7B 6.0 s per image,
InternVL3-8B 8.4 s, Gemma-3-12B 11.3 s, Llama-3.2-11B-Vision 55.4 s.

```powershell
cd evaluation
$env:MAX_EVAL = 5; ..\.venvs\vlm\Scripts\python vlm_eval.py qwen out_qwen          # base model, no adapter
..\.venvs\vlm\Scripts\python vlm_predict.py --model qwen_base --image <photo>
```
