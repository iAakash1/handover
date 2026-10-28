# 02 · LLaVA-1.6-Mistral-7B + LoRA (March 2026): 2% accuracy

**What we did:** fine-tuned `llava-hf/llava-v1.6-mistral-7b-hf` with LoRA (r 64, alpha 128, dropout 0.05 on
q/k/v/o/gate/up/down) on the PlantVillage captions. It trained from 24 March 16:36 to 25 March 09:45, about 36 s per
step and 1,700 steps. An adapter was saved every 100 steps, and `best/` was saved at 04:25 on the lowest
validation loss.

**Result (same 200 validation images as every VLM):** 2.00% accuracy, macro-F1 0.0151. The answers were long and
generic and named no disease in 178 of 200 cases. Report with every answer: `../../evaluation/llava_v16/`.

**Code:** the original March training script was not kept; only the adapter survived. `02_finetune_llava.py` here is
the training rebuilt from what the adapter itself records. Its trainable parameter count, 177,209,344, matches the
saved 708 MB fp32 adapter exactly. Two values were not recorded anywhere and are taken from our other VLM runs:
LR 2e-4, and batch 1 × grad-accum 8. A two-step run was tested on the RTX 4500.

## Run it (vlm env, from the repository root)

```powershell
# the March adapter on one photo (downloaded by setup/download_models.py)
cd evaluation
..\.venvs\vlm\Scripts\python vlm_predict.py --model llava --image <photo>
$env:MAX_EVAL = 5; ..\.venvs\vlm\Scripts\python vlm_eval.py llava out_llava ..\projects\04_llava_blip2\outputs\llava_plantvillage\best

# train again (writes outputs\llava_plantvillage_rerun here; the March adapter is never touched)
cd ..\failed_experiments\02_llava16_mistral_lora
..\..\.venvs\vlm\Scripts\python 02_finetune_llava.py --max-steps 2 --val 2   # quick check
..\..\.venvs\vlm\Scripts\python 02_finetune_llava.py                         # full epoch, ~23 h
```

The March adapter and its checkpoints are at `projects/04_llava_blip2/outputs/llava_plantvillage/`.
