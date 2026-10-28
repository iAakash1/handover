# 06 · Qwen2.5-VL-7B on PlantVillage (April 2026) — the model that worked

Qwen2.5-VL-7B-Instruct fine-tuned with LoRA (r32, alpha 64, bf16) on structured PlantVillage captions. It writes
a full diagnosis: plant, condition, severity, pathogen and visible symptoms.

**Environment:** `vlm` · **GPU:** NVIDIA, 24 GB (the base model is 16 GB in bf16)

## Results (200 PlantVillage validation images)

| Model | Accuracy | Macro-F1 | BLEU-4 | ROUGE-L | Severity R² |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-VL-7B + LoRA r32 | **98.50%** | 0.9827 | 0.4556 | 0.5869 | 0.9654 |
| Same model without fine-tuning | 8.50% | 0.0800 | – | – | – |

Report: `qwen/evaluation_report.json`; zero-shot: `../../evaluation/qwen_zeroshot`.

## Run it (from the repository root)

Quick diagnosis of one photo:

```powershell
cd evaluation
..\.venvs\vlm\Scripts\python vlm_predict.py --model qwen_april --image "C:\Users\Admin\Desktop\VLM\PlantVillage\Tomato_Early_blight\<any image>.JPG"
```

The full app (CLIP check that rejects non-leaf photos, structured answer, plant-mismatch check, GradCAM heat-map):

```powershell
cd projects\06_qwen25vl_plantvillage\crop_disease_app
..\..\..\.venvs\vlm\Scripts\python check_setup.py
..\..\..\.venvs\vlm\Scripts\python -m streamlit run app.py
```

The app finds the adapter in `../qwen/outputs/qwen25vl_plantvillage/best` by itself; other settings can go in a
`.env` file (see `.env.example`).

## Train / evaluate again (original April scripts)

```powershell
cd projects\06_qwen25vl_plantvillage\qwen
..\..\..\.venvs\vlm\Scripts\python 01_prepare_dataset.py    # captions.json (90/10 split, seed 42) from PLANTVILLAGE_DIR
..\..\..\.venvs\vlm\Scripts\python 02_finetune_qwen25vl.py  # 3 epochs, LoRA r32, LR 2e-4, batch 4 x grad-accum 4
..\..\..\.venvs\vlm\Scripts\python 04_evaluate.py           # the 200-image evaluation above
..\..\..\.venvs\vlm\Scripts\python -m streamlit run 05_app.py
```

## Files

| Path | What it is |
|---|---|
| `qwen/` | April scripts, `captions.json`, the adapter (`outputs/qwen25vl_plantvillage/best`), reports |
| `qwen_variant/` | An earlier variant of the same scripts, kept for reference |
| `crop_disease_app/` | Production app (from github.com/kharulkamaaksha/Crop_Disease_VLM, July 2026) |
