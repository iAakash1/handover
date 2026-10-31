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
`quick_eval_florence2.py` are kept as we wrote them (run from this folder with the `florence` env). The only
change: the two quick scripts read `../tp/captions.json` on the workstation, which is now
`../../projects/04_llava_blip2/blip2_rerun_june/captions.json` (the same file); the originals are in
`../../archive/original_files/workstation/VLM/florence/`. Fine-tuning did not complete, so there is no Florence
adapter.

`captions.json` here (used by `02b_finetune_florence2.py`) is the workstation file exactly as it was: it holds the
same `{"train", "val"}` document twice in a row (18,574 + 2,064 captions each), so `json.load` stops with "Extra
data". It is kept unchanged as a record; to run `02b_finetune_florence2.py`, read only the first document, e.g.
`json.JSONDecoder().raw_decode(open("captions.json").read())[0]`.
