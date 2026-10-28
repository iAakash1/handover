# 01 · Florence-2 (April and June 2026) — tried and dropped

Microsoft's Florence-2-base-ft (232 M parameters) was the third VLM we tried. Fine-tuning never finished: with
transformers 5.3 the model's remote code hit five different incompatibilities one after another
(`patch_florence2.py` fixes the first two). We moved to Qwen2.5-VL instead.

To still give it a fair number, I evaluated the base model zero-shot with the transformers version it was written
for (4.46.3): it writes scene captions ("a green leaf on the ground") and never names a disease.

| Model | Accuracy | Report |
|---|---:|---|
| Florence-2-base-ft, zero-shot, 200 PlantVillage images | 0.00% | `../../evaluation/florence_zeroshot` |

**Environment:** `florence` (same as `vlm` but transformers 4.46.3 — do **not** use the `vlm` env, the model
produces garbage under transformers 5.x)

## Run it (from the repository root)

```powershell
cd evaluation
..\.venvs\florence\Scripts\python vlm_predict.py --model florence --image <photo>
$env:MAX_EVAL = 5; ..\.venvs\florence\Scripts\python vlm_eval.py florence out_florence
```

## The original scripts

`02b_finetune_florence2.py`, `quick_train_florence2.py`, `03_inference.py`, `04_evaluate.py`, `05_app.py`,
`quick_eval_florence2.py` are kept exactly as we wrote them (run from this folder with the `florence` env).
Fine-tuning did not complete, so there is no Florence adapter.
