# Handover guide: PlantVLM, Qwen2.5-VL on expert-written captions (project 08)

**1. What it does.** Describes the visible condition of a tomato leaf, trained only on captions written and revised
by people. It is an exploratory study of whether a few high-quality captions are enough; they were not (2 of 6
validation images right), so it is also listed in [`../../failed_experiments/07_plantvlm_expert_captions/`](../../failed_experiments/07_plantvlm_expert_captions/README.md).

**2. Architecture.** `Qwen/Qwen2.5-VL-7B-Instruct` in bf16 with LoRA r16, alpha 32, dropout 0.05 on q/k/v/o and
gate/up/down projections of the language model (vision tower excluded). Code: `plantvlm/` package, `scripts/00–04`,
settings in `configs/qwen25vl_lora_tomato.yaml`.

**3. Dataset.** 30 PlantVillage tomato images, 6 diseases × 5 (bacterial spot, early blight, leaf mold, mosaic virus,
target spot, yellow leaf curl virus), each with 5 expert-revised captions (150), in
`1stJuly/Revised_captions.xlsx`; the images are in `1stJuly/Dataset/` (not in git; on the TIH workstation). Split:
grouped rotation, fold 0, 3 / 1 / 1 images per class for train / validation / test (`data/{train,val,test}.jsonl`).

**4. Training.** Up to 8 epochs, micro-batch 2 × gradient accumulation 4, LR 1e-4, weight decay 0.01, warm-up 10%,
early stopping (patience 2), seed 42 (`scripts/02_train.py`, about 1 minute per epoch). Best epoch: 3 (validation
loss 1.823).

**5. Weights.** `runs/run_002_fold0/checkpoints/best/` (adapter files in git; `adapter_model.safetensors`, 161 MB,
fetched by `python setup\download_models.py 08_plantvlm`); epochs 1–5 with `--all`.

**6. Install.** `.venvs\vlm`; NVIDIA GPU with 24 GB.

**7. Inference.**
```powershell
cd projects\08_plantvlm_expert_captions
..\..\.venvs\vlm\Scripts\python scripts/03_infer.py --split val                # fine-tuned answers
..\..\.venvs\vlm\Scripts\python scripts/03_infer.py --split val --baseline     # base model
cd ..\..\evaluation; ..\.venvs\vlm\Scripts\python vlm_predict.py --model plantvlm --image <leaf photo>
```
`03_infer.py` samples (temperature 0.7) and overwrites `runs/run_002_fold0/predictions/`; restore the published
ones with `git checkout -- runs/run_002_fold0/predictions`.

**8. Reproduce evaluation.** `..\..\.venvs\vlm\Scripts\python scripts/04_evaluate.py --split val --compare-baseline`.
Reference (`runs/run_002_fold0/evaluation/val_report.json`): right disease for 2 of 6 validation images (chance 1 of
6), BLEU-1 0.590, BLEU-4 0.045, ROUGE-L 0.262.

**9. Modify / extend.** Every setting is in the YAML config and can be overridden on the command line, e.g.
`scripts/02_train.py --set train.epochs=12 --set lora.r=8` or `scripts/01_build_dataset.py --set split.fold=2`.
More images per class is the change most likely to help.

**10. Limitations.** 30 images (6 for validation) are far too few for a 7B model; results are not significant.
