# 04 · BLIP-2 June re-run scored 0% (June 2026)

In June we re-trained BLIP-2 OPT-2.7B + LoRA for one epoch. Its own evaluation script
(`projects/04_llava_blip2/blip2_rerun_june/04_evaluate.py`) wrote **0% classification accuracy** (BLEU-1 0.2081) to
`evaluation_report.json`.

Re-scored in October with the April protocol (`evaluation/vlm_eval.py`, the same 200 images as every VLM), the same
adapter reaches **89.00%** accuracy, macro-F1 0.8948 (`evaluation/blip2_june/`), which matches the March run. So the
model was fine; the June scoring was not.

The code stays in `projects/04_llava_blip2/blip2_rerun_june/` because the model itself works. To see both numbers:

```powershell
cd evaluation
$env:MAX_EVAL = 5; ..\.venvs\vlm\Scripts\python vlm_eval.py blip2 out_blip2j ..\projects\04_llava_blip2\blip2_rerun_june\outputs\blip2_plantvillage\best
```
