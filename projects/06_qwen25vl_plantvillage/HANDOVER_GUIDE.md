# Handover guide: Qwen2.5-VL-7B with LoRA on PlantVillage, and the crop disease app (project 06)

**1. What it does.** Writes a structured diagnosis (plant, condition, severity, pathogen, symptoms) for a tomato,
potato or pepper leaf photo. The crop disease app wraps it with an out-of-distribution check, an output parser, a
plant-mismatch check and a GradCAM heat-map.

**2. Architecture.** `Qwen/Qwen2.5-VL-7B-Instruct` in bf16 with a LoRA adapter (r32, alpha 64) on all attention and MLP
projections. App pipeline (`crop_disease_app/pipeline.py`): (1) CLIP ViT-B/32 similarity to leaf prompts, stop below
0.27; (2) Qwen generates the caption (beam search 4); (3) `utils/parser.py` extracts the fields and stops if critical
fields are missing; (4) `utils/validator.py` stops if the detected plant differs from the one selected; (5) the
confidence shown is derived from the CLIP similarity. `app.py` adds a GradCAM heat-map (`utils/gradcam.py`).

**3. Dataset.** PlantVillage, 15 classes (tomato, potato, pepper; `PLANTVILLAGE_DIR`) with structured captions in
`qwen/captions.json`: 90/10 split, seed 42 (18,574 train / 2,064 validation records); the evaluation uses the first
200 validation records.

**4. Training.** 3 epochs, LR 2e-4, batch 4 × gradient accumulation 4, bf16 (`qwen/02_finetune_qwen25vl.py`). The
first attempt needed four fixes, kept in [`../../failed_experiments/05_qwen_april_first_attempt/`](../../failed_experiments/05_qwen_april_first_attempt/README.md).

**5. Weights.** `qwen/outputs/qwen25vl_plantvillage/best/` (adapter files in git; `adapter_model.safetensors`,
415 MB, fetched by `python setup\download_models.py 06_qwen25vl`). The app finds it by default
(`VLM_ADAPTER_PATH` overrides).

**6. Install.** `.venvs\vlm`; NVIDIA GPU with ~16 GB+; the base model (~16 GB) and CLIP download from Hugging Face on
first use.

**7. Inference.**
```powershell
python run_app.py 2                                        # the crop disease app (Streamlit)
cd evaluation
..\.venvs\vlm\Scripts\python vlm_predict.py --model qwen_april --image <leaf photo>
```
`crop_disease_app\check_setup.py` checks the app's model paths and packages before the first start.

**8. Reproduce evaluation.**
```powershell
cd projects\06_qwen25vl_plantvillage\qwen
..\..\..\.venvs\vlm\Scripts\python 04_evaluate.py          # the 200-image evaluation
```
Reference (`qwen/evaluation_report.json`): **98.50%** accuracy, macro-F1 0.9827, BLEU-4 0.4556, ROUGE-L 0.5869,
severity R² 0.9654; the same model without fine-tuning 8.50% ([`../../evaluation/qwen_zeroshot/`](../../evaluation/qwen_zeroshot/)).

**9. Modify / extend.** App settings are in `crop_disease_app/config.py` and can be set by environment variables or
`.env`: `VLM_OOD_THRESHOLD`, `VLM_ADAPTER_PATH`, `VLM_BASE_MODEL_PATH`, `VLM_MAX_NEW_TOKENS`, `VLM_NUM_BEAMS`, …;
the supported plants are `SUPPORTED_PLANTS`. Retraining: `qwen/01_prepare_dataset.py` (captions) then
`qwen/02_finetune_qwen25vl.py`. A new crop needs its class folders, caption templates and a retrain.

**10. Limitations.** Trained and evaluated on PlantVillage lab photos only. The CLIP threshold (0.27) was set by hand.
Earlier versions (the 13 Apr app, an alternative script set) are in
[`../../archive/previous_versions/06_qwen25vl_plantvillage/`](../../archive/previous_versions/06_qwen25vl_plantvillage/README.md).
