# Handover guide: BLIP-2 and LLaVA with LoRA (project 04)

**1. What it does.** Writes a structured diagnosis for a PlantVillage leaf photo ("Plant: Tomato. Condition: Late
Blight. Severity: Moderate."). BLIP-2 is the working model of this project; the LLaVA-1.6 adapter is kept for
comparison (it did not learn the task).

**2. Architecture.**

| Model | Base | Adapter |
|---|---|---|
| BLIP-2, March and June | `Salesforce/blip2-opt-2.7b` | LoRA r32, alpha 64, dropout 0.05 |
| LLaVA-1.6 (past experiment) | `llava-hf/llava-v1.6-mistral-7b-hf` | LoRA r64, alpha 128, dropout 0.05, all 7 projections |

**3. Dataset.** PlantVillage (`PLANTVILLAGE_DIR`), template captions per class, 90/10 split; evaluation on the shared
200 validation images ([`../../docs/HANDOVER_REPORT.md`](../../docs/HANDOVER_REPORT.md#5-evaluation-methodology)).

**4. Training.** March: `blip2_march/02_finetune_blip2.py` (5 epochs, 5,800 steps). June: a one-epoch re-run in
`blip2_rerun_june/`. Hyperparameters are the constants at the top of each `02_finetune_blip2.py`. LLaVA: 1,700 steps
on 24–25 Mar; its original script was not kept, a rebuilt one is in
[`../../failed_experiments/02_llava16_mistral_lora/`](../../failed_experiments/02_llava16_mistral_lora/README.md).

**5. Weights.** March BLIP-2 `outputs/blip2_plantvillage/best/`, June BLIP-2
`blip2_rerun_june/outputs/blip2_plantvillage/best/`, LLaVA `outputs/llava_plantvillage/best/` (adapter files in git,
`adapter_model.safetensors` fetched by `python setup\download_models.py 04_llava`).

**6. Install.** `.venvs\vlm` (PyTorch 2.5.1 CUDA 12.1, transformers 5.3, PEFT 0.18); NVIDIA GPU with 8 GB+ (BLIP-2)
or 16 GB+ (LLaVA). The base models download from Hugging Face on first use (BLIP-2 ~14 GB, LLaVA ~15 GB).

**7. Inference.**
```powershell
cd evaluation
..\.venvs\vlm\Scripts\python vlm_predict.py --model blip2_march --image <leaf photo>   # or blip2_june, llava
python ..\run_app.py 6                                                                # June demo app (7 = March)
```

**8. Reproduce evaluation.**
```powershell
cd evaluation
$env:MAX_EVAL = 5      # optional quick check; unset = all 200 images
..\.venvs\vlm\Scripts\python vlm_eval.py blip2 out_blip2 ..\projects\04_llava_blip2\outputs\blip2_plantvillage\best
```
Reference: BLIP-2 March **89.00%** (macro-F1 0.8928, [`../../evaluation/blip2_march/`](../../evaluation/blip2_march/)), June 89.00% (0.8948,
[`blip2_june`](../../evaluation/blip2_june/)), LLaVA-1.6 2.00% ([`llava_v16`](../../evaluation/llava_v16/)).

**9. Modify / extend.** Caption templates live in `01_prepare_dataset.py`; training settings at the top of
`02_finetune_blip2.py`; generation settings (beams, penalties) in `03_inference.py` and `evaluation/vlm_eval.py`.

**10. Limitations.** BLIP-2 writes short captions and was superseded by Qwen2.5-VL (project 06, 98.50%). The June
script's own evaluation reported 0% because of a scoring problem
([`../../failed_experiments/04_blip2_june_zero_score/`](../../failed_experiments/04_blip2_june_zero_score/README.md)). LLaVA-1.6 answers generically (2%).
