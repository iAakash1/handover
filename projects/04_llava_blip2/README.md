# 04 · LLaVA-1.6 and BLIP-2 with LoRA (March and June 2026)

My first vision-language models. The goal: instead of only a label, write a structured diagnosis for a leaf
photo ("Plant: Tomato. Condition: Late Blight. Severity: Moderate..."). Both models were fine-tuned with LoRA on
PlantVillage captions; Qwen2.5-VL (project 06) replaced them in April.

**Environment:** `vlm` · **GPU:** NVIDIA, 16 GB+ (LLaVA uses ~16 GB in bf16, BLIP-2 ~8 GB)

## Results (same 200 PlantVillage validation images for every VLM)

| Model | Accuracy | Macro-F1 | Notes |
|---|---:|---:|---|
| BLIP-2 OPT-2.7B + LoRA r32, March (5,800 steps) | **89.00%** | 0.8928 | short, correctly structured answers |
| BLIP-2 OPT-2.7B + LoRA r32, June 1-epoch re-run | 89.00% | 0.8948 | its own June script recorded 0% — a scoring bug, not the model |
| LLaVA-1.6-Mistral-7B + LoRA r64, March (1,700 steps) | 2.00% | 0.0151 | long generic answers, named no disease in 178 of 200 |
| LLaVA-1.5-7B | – | – | training stopped by a batch-collation error (PoCRA report) |

Reports with every generated answer: `../../evaluation/blip2_march`, `blip2_june`, `llava_v16`.

## Run it (from the repository root)

```powershell
cd evaluation
..\.venvs\vlm\Scripts\python vlm_predict.py --model blip2_march --image "C:\Users\Admin\Desktop\VLM\PlantVillage\Tomato_Late_blight\<any image>.JPG"
..\.venvs\vlm\Scripts\python vlm_predict.py --model llava --image <photo>
$env:MAX_EVAL = 5; ..\.venvs\vlm\Scripts\python vlm_eval.py blip2 out_blip2 ..\projects\04_llava_blip2\outputs\blip2_plantvillage\best
```

(`MAX_EVAL` unset = the full 200 images.)

The June BLIP-2 scripts are a complete pipeline (`blip2_rerun_june/`, run from that folder):

```powershell
cd projects\04_llava_blip2\blip2_rerun_june
..\..\..\.venvs\vlm\Scripts\python 03_inference.py              # one image per PlantVillage class
..\..\..\.venvs\vlm\Scripts\python -m streamlit run 05_app.py   # upload-a-photo demo app
```

## Train again

```powershell
cd projects\04_llava_blip2\blip2_rerun_june
..\..\..\.venvs\vlm\Scripts\python 01_prepare_dataset.py   # writes captions.json from PLANTVILLAGE_DIR
..\..\..\.venvs\vlm\Scripts\python 02_finetune_blip2.py    # LoRA r32, writes outputs\blip2_plantvillage\best
..\..\..\.venvs\vlm\Scripts\python 04_evaluate.py          # BLEU / ROUGE / CLIP + classification
```

There is no separate LLaVA training script in our folders; only its trained adapter survived
(`outputs/llava_plantvillage/best`).

## Files

| Path | What it is |
|---|---|
| `outputs/blip2_plantvillage/best` | March BLIP-2 adapter |
| `outputs/llava_plantvillage/best` | March LLaVA-1.6 adapter |
| `blip2_rerun_june/` | June BLIP-2 code, captions, its adapter (`outputs/…/best`) and its own reports |
