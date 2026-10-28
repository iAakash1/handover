# Earlier experiments on the Mac (12 June – 1 July 2026)

Everything tried on the Mac before the knowledge-grounded pipeline in the parent folder, in the order it was done.
Together they explain why the July pipeline looks the way it does: off-the-shelf VLMs could not name the disease
(03, 04), so the captions had to be anchored to the known label and a knowledge base (01, 02), which became the
cited Disease Knowledge Base of July.

| # | Folder | Date | What | Result |
|---|---|---|---|---|
| 01 | `01_plantcaption_pipeline_2026-06-12/` | 12 Jun | **PlantCaption**: 5 captions per image from two parts, agronomy from a knowledge base anchored to the label (`agronomy_kb.py`) and the photo's scene from Qwen2.5-VL (`visual_analyzer.py`), fused by `caption_synthesizer.py`; Streamlit review console | sample outputs for 20 images in `sample_outputs/` |
| 02 | `02_caption_sets_2026-06-23/` | 23 Jun | Four successive caption sets, 5 captions per image: `a_` for 20 TIH sample photos (10 crops), then `b_` → `c_` → `d_` for the 16 benchmark leaves, each round adding quality control, rewriting and a voice-diversity audit | `d_16_images_final_2203/captions.json` (80 captions) is the final set |
| 03 | `03_zero_shot_4vlm_benchmark_2026-06-25/` | 24–25 Jun | **Zero-shot benchmark, 4 VLMs on MLX** (Qwen2.5-VL-7B, Gemma-3-12B, InternVL3-8B, Llama-3.2-11B-Vision, all 4-bit) on 16 leaf images, one fixed prompt | exact disease match 0–6% per model, right disease mentioned anywhere 19–50%; Qwen2.5-VL fastest (6.0 s/image) |
| 04 | `04_zero_shot_7vlm_benchmark_2026-07-01/` | 1 Jul | **Zero-shot benchmark, 7 VLMs on MLX** (the four above + Qwen3-VL-8B, Molmo-7B-D, DeepSeek-VL2-small) on 30 tomato leaves, 6 diseases × 5; the same 30 images and 150 expert captions as project 08 | best accuracy 23% (InternVL3-8B); Qwen3-VL-8B degenerate on 16 of 30 images |

The base-model rows of the July evaluation are a third zero-shot measurement, on the full test sets: Qwen2.5-VL-7B
4-bit gets 7.3% on 910 tomato and 17.5% on 200 mango images (`../reports/qwen25vl_<crop>_qlora/evaluation/`).
No one-shot or few-shot (in-context example) experiment was run on the Mac.

## How to run

All four need macOS on Apple Silicon for the MLX parts; use a Python 3.12 env (`conda create -n vlm python=3.12`).

```bash
# 01 PlantCaption (transformers backend; "mock" needs no model)
cd earlier_experiments/01_plantcaption_pipeline_2026-06-12/captioning
pip install -r requirements.txt
python make_sample_images.py              # synthetic test images into ../images
python run_pipeline.py --backend mock     # plumbing check, seconds
python run_pipeline.py --backend qwen     # real run with Qwen2.5-VL-7B
streamlit run review_app.py               # review, edit, export

# 03 four-model benchmark: benchmark -> ground truth -> (LLM judge) -> evaluate -> plots -> report
cd earlier_experiments/03_zero_shot_4vlm_benchmark_2026-06-25
pip install -r requirements.txt
bash run_all.sh

# 04 seven-model benchmark (collects predictions; scoring is in report/)
cd earlier_experiments/04_zero_shot_7vlm_benchmark_2026-07-01
pip install -r requirements.txt
python scripts/run_benchmark.py
python scripts/run_benchmark.py --models mlx-community/Qwen2.5-VL-7B-Instruct-4bit   # one model
```

The models download from Hugging Face (`mlx-community/...`, 4-bit) on first use. The LLM-judge step of 03 needs an
`OPENAI_API_KEY` or `GEMINI_API_KEY`; none was set on the Mac, so the report has no judge scores.

## What is in each folder

- **01:** `captioning/` (all code, its own README), `sample_outputs/` (`captions_master.SAMPLE.json`,
  `captions_review.SAMPLE.csv`, `train.SAMPLE.jsonl`).
- **02:** per version `captions.json` / `captions.csv`, `caption_statistics.txt`, `validation_report.txt`,
  duplicate and phrase-frequency reports, `logs/` (generation, quality control, rewrite, voice diversity). The 16
  images are `../03_zero_shot_4vlm_benchmark_2026-06-25/images/` (identical files). The 20 TIH sample photos of
  `a_` are TIH-provided and not in git.
- **03:** `scripts/` (benchmark, ground truth from file names, LLM judge, evaluation, plots, report), `images/`,
  raw answers per model (`results/<model>/*.txt`, `json_results/`), [`report/final_report.md`](03_zero_shot_4vlm_benchmark_2026-06-25/report/final_report.md) + PDF + charts (`report/charts/`).
- **04:** `scripts/` (runner, parser, memory/metadata helpers, validation, aggregation), `dataset/` (30 images),
  every prediction (`outputs/predictions_<model>.json`, `results/predictions.csv`), per-model logs,
  `report/PlantDx_VLM_Evaluation_Report.docx` (also as [`report/PlantDx_VLM_Evaluation_Report.md`](04_zero_shot_7vlm_benchmark_2026-07-01/report/PlantDx_VLM_Evaluation_Report.md)
  with its 14 charts in `report/figures/`, readable on GitHub).
