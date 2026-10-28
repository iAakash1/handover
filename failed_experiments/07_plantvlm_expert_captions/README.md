# 07 · PlantVLM on expert captions only (August 2026)

We fine-tuned Qwen2.5-VL + LoRA r16 on 150 hand-written captions, revised by an expert, for 30 images of 6 diseases.
The best epoch (3) named the right disease for **2 of 6** validation images; guessing would get 1 of 6. The captions
were good, but 30 images are far too few.

The code still runs, so it stays in `projects/08_plantvlm_expert_captions/` (README there; report
`runs/run_002_fold0/evaluation/val_report.json`).
