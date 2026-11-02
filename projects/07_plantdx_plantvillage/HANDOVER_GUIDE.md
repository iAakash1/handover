# Handover guide: PlantDx, Qwen2.5-VL on 15 PlantVillage classes (project 07)

**1. What it does.** Writes a structured diagnosis for tomato, potato and pepper leaf photos (15 PlantVillage
classes). It is the April pipeline of project 06 rebuilt as a modular project with run tracking and an app.

**2. Architecture.** `Qwen/Qwen2.5-VL-7B-Instruct` + LoRA (r32, alpha 64). Code: `core/dataset.py` (captioned
image dataset), `core/trainer.py` (training loop), `core/inference_engine.py` (generation), `core/run_manager.py`
(numbered runs in `outputs/runs/`), `utils/` (paths, model loading, metrics, IO), `scripts/01–04`, `app/05_app.py`
(Streamlit, CLIP-based photo check).

**3. Dataset.** PlantVillage, 15 class folders (`PLANTVILLAGE_DIR`); captions and the 90/10 split (seed 42) are
built by `scripts/01_prepare_dataset.py` into `data/` (split file `data/splits/splits.json`).

**4. Training.** `configs/config.yaml`: LoRA r32 / alpha 64, 3 epochs, batch 4 × gradient accumulation 4, LR 2e-4,
max length 512 (`scripts/02_train.py`; about 2.5 h per epoch on the RTX 4500 Ada).

**5. Weights.** `outputs/checkpoints/best/` (adapter files in git; `adapter_model.safetensors`, 381 MB, fetched by
`python setup\download_models.py 07_plantdx`).

**6. Install.** `.venvs\vlm`; NVIDIA GPU with ~16 GB+; the base model downloads from Hugging Face on first use.

**7. Inference.**
```powershell
python run_app.py 4                                        # PlantDx Streamlit app
cd evaluation
..\.venvs\vlm\Scripts\python vlm_predict.py --model plantdx --image <leaf photo>
```

**8. Reproduce evaluation.**
```powershell
cd projects\07_plantdx_plantvillage
..\..\.venvs\vlm\Scripts\python scripts/04_evaluate.py     # 200 validation images (evaluation.max_samples)
```
Reference (`outputs/evaluations/report.json`): **98.00%**, macro-F1 0.9777, BLEU-4 0.386, severity R² 0.966.

**9. Modify / extend.** Everything is in `configs/config.yaml`: `paths`, `model`, `lora`, `training`, `inference`
(e.g. `max_new_tokens`, `repetition_penalty`), `evaluation` (e.g. `max_samples`, CLIP model), `app`. New classes:
add their folders under `PLANTVILLAGE_DIR` and their caption templates in `scripts/01_prepare_dataset.py`, then
re-run 01 → 02 → 04.

**10. Limitations.** PlantVillage lab photos only. The first restructured version (15 May) is kept in
[`../../archive/previous_versions/07_plantdx_plantvillage/`](../../archive/previous_versions/07_plantdx_plantvillage/README.md).
