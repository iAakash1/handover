# How to run, retrain and extend the models

## 1. One-time setup (Windows, PowerShell, inside the repository folder)

| Step | Command | Notes |
|---|---|---|
| Python | install Python 3.10 (64-bit) from python.org, tick "Add to PATH" | the environments were built with 3.10.8 |
| Environments | `python setup\setup_envs.py` | creates `.venvs\keras`, `soynet`, `vlm`, `florence`, `tih` (~25 GB, 20–40 min) |
| Models | `python setup\download_models.py` | 3.6 GB from github.com/iAakash1/TIH-IoT, checksummed, re-runnable (`--all` adds older versions and per-epoch checkpoints, +3.4 GB) |
| Check | `python setup\check_setup.py` | must print **READY** |

GPU: an NVIDIA card with a recent driver. The VLMs (04–08, 10) need ~16–24 GB of GPU memory; the CNNs (01–03)
also run on CPU.

## 2. Run any model

**Easiest:** `python run_app.py` and pick a number. 1 = TIH tomato (ICAIN paper model), 2 = crop disease app
(CLIP check + GradCAM), 3 = every model on one page. It starts Streamlit with the right environment and prints the
address to open (normally http://localhost:8501).

From the command line:

Always `cd` into the project folder first; every README shows the exact command. The pattern is:

```powershell
cd projects\<project>
..\..\.venvs\<env>\Scripts\python <script> [options]
```

| I want to… | Command (from the repository root) |
|---|---|
| Classify a tomato leaf with a CNN | `cd projects\01_cnn_keras_tomato; ..\..\.venvs\keras\Scripts\python predict.py --image sample.jpg` |
| Same, best CNN (PyTorch) | `cd projects\02_cnn_pytorch_convnext; ..\..\.venvs\vlm\Scripts\python predict.py --image <photo>` |
| Soybean healthy vs diseased | `cd projects\03_soybean_soynet; ..\..\.venvs\soynet\Scripts\python predict_soynet.py --image <photo>` |
| Get a written diagnosis from a VLM | `cd evaluation; ..\.venvs\vlm\Scripts\python vlm_predict.py --model qwen_april --image <photo>` |
| Open the diagnosis web app | `python run_app.py 2` (or `cd projects\06_qwen25vl_plantvillage\crop_disease_app; ..\..\..\.venvs\vlm\Scripts\python -m streamlit run app.py`) |
| Open the TIH tomato app | `python run_app.py 1` (or `cd projects\10_tih_tomato_qwen_qlora; ..\..\.venvs\tih\Scripts\python -m streamlit run streamlit_app.py`) |
| Try every model on one photo | `python run_app.py 3` |
| TIH tomato (11 classes) | `cd projects\10_tih_tomato_qwen_qlora; ..\..\.venvs\tih\Scripts\python -m plantdx.cli company predict --config configs/train/qwen25vl_tih_tomato_moreaccurate.yaml --adapter checkpoints/tih_tomato_moreaccurate --image <photo>` |
| Re-create an evaluation number | each project has `evaluate.py` / `04_evaluate.py`; VLMs: `cd evaluation; ..\.venvs\vlm\Scripts\python vlm_eval.py <kind> <out_dir> <adapter>` |

`vlm_predict.py --model` accepts `qwen_april`, `plantdx`, `plantvlm`, `qwen_base`, `blip2_march`, `blip2_june`,
`llava` (vlm env) and `florence` (florence env).

## 3. Retrain

| Project | Command | Measured time (from our logs) |
|---|---|---|
| 01 Keras | move the `.keras` file out of `model\`, run `main_convnext.py` (or `main_mobilenet.py`, `main.py`) | not logged (trained on a Mac) |
| 02 ConvNeXt | `notebooks\02_training_convnext.ipynb` | ~9 min (10 epochs × 0.9 min) |
| 03 SoyNet | `run_pytorch.py` (all four) or `run_experiments.py` (Keras) | Keras ResNet50 261 s, MobileNetV2 166 s |
| 04 BLIP-2 | `blip2_rerun_june\02_finetune_blip2.py` | not logged |
| 06 Qwen2.5-VL | `qwen\01_prepare_dataset.py` then `qwen\02_finetune_qwen25vl.py` | not logged |
| 07 PlantDx | `scripts\01_prepare_dataset.py`, `scripts\02_train.py` | ~2.5 h per epoch |
| 08 PlantVLM | `scripts\02_train.py` | ~1 min per epoch |
| 10 TIH tomato | `-m plantdx.cli company preflight`, then `company train` (same `--config`) | 193 min (2.9 epochs) |

New models are written next to the old ones (same folder names), so move the old files away first if you want
to keep them.

## 4. Where to change things

| Change | Where |
|---|---|
| Dataset location | environment variables in `docs/datasets.md` (e.g. `$env:PLANTVILLAGE_DIR = "E:\data\PlantVillage"`) |
| CNN hyper-parameters | `projects/01_…/model/train_*.py`, `projects/03_…/config.py`, `pytorch/config.py` |
| VLM prompts and generation | `crop_disease_app/config.py`, `07_…/configs/config.yaml`, `08_…/configs/*.yaml` |
| TIH tomato training | `projects/10_…/configs/train/qwen25vl_tih_tomato_moreaccurate.yaml` (LoRA rank, LR, image size, epochs) |
| TIH tomato classes | `projects/10_…/configs/company_tomato_classes.yaml` |
| A new crop for the TIH pipeline | copy `company_tomato_classes.yaml`, list the new class folders, point a copy of the train config at it |

## 5. Common problems

| Message | Fix |
|---|---|
| `MISSING` in `check_setup.py` | run the command printed next to it |
| `adapter not found` / `No such file … .safetensors` | `python setup\download_models.py` |
| `CUDA out of memory` | close other GPU programs; only one 7B VLM fits in 24 GB at a time |
| Florence-2 prints nonsense words | you used the `vlm` env; use `.venvs\florence` |
| `torch.load` / CLIP safetensors error | you are offline on first use; connect once so Hugging Face can download the base models |
| TIH: `dataset root not found` | add `--dataset-root <folder with the 11 class folders>` |
| Wrong folder errors (`No module named …`) | `cd` into the project folder before running |
