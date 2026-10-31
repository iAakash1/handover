# Model card: zero-shot VLM benchmarks on MLX (no training)

Off-the-shelf vision-language models, run on a Mac with MLX and asked to describe leaf photos with no fine-tuning.
Three measurements; the first two are self-contained folders with their own code and full README.

| Benchmark | Models (all `mlx-community/*-4bit`) | Images | Code and full README | Result |
|---|---|---|---|---|
| 4-VLM, 24–25 Jun 2026 | Qwen2.5-VL-7B-Instruct, gemma-3-12b-it, InternVL3-8B, Llama-3.2-11B-Vision-Instruct | 16 leaves (tomato, potato, pepper, cotton) in `images/` | [`earlier_experiments/03_zero_shot_4vlm_benchmark_2026-06-25/`](../../earlier_experiments/03_zero_shot_4vlm_benchmark_2026-06-25/README.md) | exact disease match 0–6% per model; right disease mentioned anywhere 19–50%; Qwen2.5-VL fastest, 6.0 s/image |
| 7-VLM, 1 Jul 2026 | the four above + Qwen3-VL-8B-Instruct, Molmo-7B-D-0924, deepseek-vl2-small | 30 tomato leaves, 6 diseases × 5, in `dataset/` | [`earlier_experiments/04_zero_shot_7vlm_benchmark_2026-07-01/`](../../earlier_experiments/04_zero_shot_7vlm_benchmark_2026-07-01/README.md) | best accuracy 23% (InternVL3-8B); Qwen3-VL-8B degenerate on 16 of 30 |
| Base rows of the July evaluation | Qwen2.5-VL-7B-Instruct | 910 tomato / 200 mango test images | `plantdx evaluate` (the fine-tuned model cards) | 7.3% / 17.5% |

## Purpose

Show whether a general VLM can name plant diseases before investing in fine-tuning. It cannot, which is why the
July pipeline fine-tunes on knowledge-grounded captions.

## Environment

macOS on Apple Silicon, Python 3.12 (conda env `vlm`); each folder has its own `requirements.txt`
(`mlx`, `mlx-vlm`, Pillow, plus plotting and report libraries for the 4-VLM suite). Each model is downloaded from
Hugging Face on first use (4–8 GB each in 4-bit). The 4-VLM suite's optional LLM-judge step needs
`OPENAI_API_KEY` or `GEMINI_API_KEY`; none was set in June, so its report has no judge scores.

## Run

```bash
# 4-VLM: benchmark -> ground truth from file names -> (optional judge) -> evaluate -> CSV -> plots -> report
cd earlier_experiments/03_zero_shot_4vlm_benchmark_2026-06-25
pip install -r requirements.txt
bash run_all.sh                      # writes json_results/, results/, csv/, plots/, reports/final_report.md

# 7-VLM: collect predictions (scoring is in report/)
cd earlier_experiments/04_zero_shot_7vlm_benchmark_2026-07-01
pip install -r requirements.txt
python scripts/run_benchmark.py --limit 1          # smoke test: one image per class
python scripts/run_benchmark.py                    # all 7 models x 30 images
python scripts/run_benchmark.py --models mlx-community/Qwen2.5-VL-7B-Instruct-4bit
```

## Inputs, outputs and results in git

- 4-VLM: `images/` (16), raw answers `results/<model>/*.txt` and `json_results/`, and the June report in `report/`
  (`final_report.md`, `crop_vlm_benchmark_report.pdf`, `charts/`, `qual_images/`). `csv/`, `plots/` and `reports/`
  are the scripts' output folders and are empty in git; a re-run fills them, so the June report in `report/` is
  never overwritten.
- 7-VLM: `dataset/` (30 images; the same images and 150 expert captions as project 08), every answer
  `outputs/predictions_<model>.json`, `results/predictions.csv`, logs, and the scored report
  `report/PlantDx_VLM_Evaluation_Report.md` (+ `.docx`, 14 figures).

## Limitations

Apple Silicon only. Small image sets (16 and 30). The prediction files contain the Mac's absolute image paths
(`/Users/aakashjawle/...`); they are records of the run and are not read back by any script.
