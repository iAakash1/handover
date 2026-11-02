# Where to start

A map of the repository for someone new to the project.

## Read first

1. [`HANDOVER_REPORT.md`](HANDOVER_REPORT.md): every model, its method, data, results (with report paths), weights,
   papers, limitations and future work.
2. The `HANDOVER_GUIDE.md` of the model you will work on (table below): what it does, how to install, run,
   evaluate and extend it, on one or two pages.
3. [`HOW_TO_RUN.md`](HOW_TO_RUN.md) for setup and every command; [`datasets.md`](datasets.md) for the data.

## Get it running (Windows workstation with an NVIDIA GPU, PowerShell)

```powershell
git clone https://github.com/iAakash1/handover.git
cd handover
python setup\setup_envs.py          # 5 environments in .venvs\ (Python 3.10)
python setup\download_models.py     # trained weights from github.com/iAakash1/TIH-IoT, SHA-256 checked
python setup\check_setup.py         # READY, or what is missing
python run_app.py                   # the Streamlit apps
```

The MLX models (project 09) need a Mac with Apple Silicon: [its guide](../projects/09_plantdx_mlx_knowledge_grounded/HANDOVER_GUIDE.md).

## Models

| Project | Model | Guide | Code | Weights | Results |
|---|---|---|---|---|---|
| 01 | Keras CNNs, tomato | [guide](../projects/01_cnn_keras_tomato/HANDOVER_GUIDE.md) | [`projects/01_cnn_keras_tomato`](../projects/01_cnn_keras_tomato/README.md) | `model/*.keras` | [`evaluation/keras/`](../evaluation/keras/README.md) |
| 02 | ConvNeXt-Tiny, PyTorch | [guide](../projects/02_cnn_pytorch_convnext/HANDOVER_GUIDE.md) | [`projects/02_cnn_pytorch_convnext`](../projects/02_cnn_pytorch_convnext/README.md) | `models/convnext_tomato_best.pth` | [`evaluation/convnext_torch/`](../evaluation/convnext_torch/README.md) |
| 03 | SoyNet, soybean | [guide](../projects/03_soybean_soynet/HANDOVER_GUIDE.md) | [`projects/03_soybean_soynet`](../projects/03_soybean_soynet/README.md) | `results/experiments/*/model.pth`, `saved_models/` | `results/experiments/*/metrics.json` |
| 04 | BLIP-2, LLaVA (LoRA) | [guide](../projects/04_llava_blip2/HANDOVER_GUIDE.md) | [`projects/04_llava_blip2`](../projects/04_llava_blip2/README.md) | `outputs/*/best`, `blip2_rerun_june/outputs/*/best` | `evaluation/blip2_march`, `blip2_june`, `llava_v16` |
| 06 | Qwen2.5-VL LoRA + crop disease app | [guide](../projects/06_qwen25vl_plantvillage/HANDOVER_GUIDE.md) | [`projects/06_qwen25vl_plantvillage`](../projects/06_qwen25vl_plantvillage/README.md) | `qwen/outputs/qwen25vl_plantvillage/best` | `qwen/evaluation_report.json` |
| 07 | PlantDx, 15 classes | [guide](../projects/07_plantdx_plantvillage/HANDOVER_GUIDE.md) | [`projects/07_plantdx_plantvillage`](../projects/07_plantdx_plantvillage/README.md) | `outputs/checkpoints/best` | `outputs/evaluations/report.json` |
| 08 | PlantVLM, expert captions | [guide](../projects/08_plantvlm_expert_captions/HANDOVER_GUIDE.md) | [`projects/08_plantvlm_expert_captions`](../projects/08_plantvlm_expert_captions/README.md) | `runs/run_002_fold0/checkpoints/best` | `runs/run_002_fold0/evaluation/val_report.json` |
| 09 | MLX tomato and mango QLoRA | [guide](../projects/09_plantdx_mlx_knowledge_grounded/HANDOVER_GUIDE.md) | [`projects/09_plantdx_mlx_knowledge_grounded`](../projects/09_plantdx_mlx_knowledge_grounded/README.md) | `checkpoints/qwen25vl_<crop>_qlora/adapters.safetensors` | `reports/` (July), `retrained_2026-11/` (Nov) |
| 10 | TIH tomato QLoRA (ICAIN 2026) | [guide](../projects/10_tih_tomato_qwen_qlora/HANDOVER_GUIDE.md) | [`projects/10_tih_tomato_qwen_qlora`](../projects/10_tih_tomato_qwen_qlora/README.md) | `checkpoints/tih_tomato_moreaccurate` | `reports/` |

Weight paths are inside the project folder. None of the weight files is in git: `setup/download_models.py` puts them
there (list and checksums: [`setup/models_manifest.json`](../setup/models_manifest.json)).

## The rest of the repository

| Folder | What it holds |
|---|---|
| [`evaluation/`](../evaluation/README.md) | Shared VLM scripts (`vlm_eval.py`, `vlm_predict.py`) and the evaluation report of every model |
| [`failed_experiments/`](../failed_experiments/README.md) | Experiments that did not work and earlier approaches, with code, results and lessons |
| [`archive/`](../archive/README.md) | Earlier versions of the final projects, and originals of files edited for the handover |
| [`history/`](../history/README.md) | Every commit, branch and tag of our code repositories (offline git bundles) |
| [`setup/`](../setup/) | Environment builder, weight downloader, setup check, repository check |
| [`docs/publication/`](publication/) | The ICAIN 2026 camera-ready paper and the paper drafts ([register](publication.md)) |
| [`docs/reports/`](reports/) | Fellowship, PoCRA and completion reports, presentations |
| [`docs/monthly/`](monthly/) | One page per month, December 2025 – October 2026 |
| [`docs/REPORT_CHECKLIST.md`](REPORT_CHECKLIST.md) | Each entry of the fellowship report and where it is here |
