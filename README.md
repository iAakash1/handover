# Handover: crop disease detection with CNNs and Vision-Language Models

TIH IIT Bombay · Chanakya Fellowship (Agri-AI 2025) · December 2025 – September 2026  
Team: Aakash Trimbak Jawle, Kamaaksha Rajendra Kharul · Faculty mentor: Dr. Santwana Gudadhe (PCCOE Pune)  
TIH mentors: Harish Lokawad, Hrishikesh Patil, Rutuja Chavan

Everything we built, trained and measured during the fellowship: code, trained models, evaluation reports,
datasets used, our ICAIN 2026 paper, and what did **not** work. Every folder has its own README with exact commands.

## Start here (Windows PowerShell, from this folder)

```powershell
python setup\setup_envs.py          # 1. creates the 5 Python environments in .venvs\ (Python 3.10, ~25 GB, once)
python setup\download_models.py     # 2. downloads all trained models (~3.6 GB) into the right folders
python setup\check_setup.py         # 3. must end with READY (lists anything missing and how to fix it)
python run_app.py                   # 4. pick an app from the menu; it opens in the browser
```

On the TIH workstation the datasets are already in place. Elsewhere, see `docs/datasets.md`.

## ⭐ Try the latest models (Streamlit)

`python run_app.py` shows this menu and starts the app with the right environment (or `python run_app.py 1`):

| # | App | What you get |
|---|---|---|
| **1** | **TIH tomato, 11 classes** · Qwen2.5-VL-7B QLoRA, Sep 2026 (**ICAIN 2026 paper**, 92.76%) | Upload tomato leaf or fruit photos → disease/pest, confidence, history; pages for training, the adapter and evaluation |
| **2** | **Crop disease app** · Qwen2.5-VL LoRA, Apr 2026 (98.50%) | CLIP check rejects non-leaf photos, full diagnosis (plant, condition, severity, pathogen, symptoms), GradCAM heat-map |
| **3** | **All models on one page** | Pick any model (CNNs, SoyNet, every VLM, the ones that failed) and run it on your photo |
| 4 | PlantDx, 15 PlantVillage classes · May 2026 (98.00%) | Tomato, potato, pepper |
| 5 | Qwen2.5-VL April demo | The first app around the 98.50% model |
| 6, 7 | BLIP-2 June and March demos (89.00%) | Caption-style diagnoses |

The first prediction of a 7B VLM takes about a minute (loading); after that a few seconds per photo. The first
ever run downloads the base model from Hugging Face (Qwen2.5-VL 16 GB, BLIP-2 14 GB).

## What is here, model by model

| # | Folder | Model | When | Best result | Env |
|---|---|---|---|---|---|
| 01 | `projects/01_cnn_keras_tomato` | Custom CNN, MobileNetV2, EfficientNetB3, ConvNeXt-Tiny (Keras) | Dec 2025 – Jan 2026 | ConvNeXt 96.15% (val) | keras |
| 02 | `projects/02_cnn_pytorch_convnext` | ConvNeXt-Tiny (PyTorch) | Jan 2026 | **99.92% test** | vlm |
| 03 | `projects/03_soybean_soynet` | ResNet50, EfficientNetB0/B3, MobileNetV2 (soybean) | Jan 2026 | ResNet50 95.16% | soynet |
| 04 | `projects/04_llava_blip2` | BLIP-2 + LoRA (March and June), LLaVA-1.6 adapter | Mar, Jun 2026 | BLIP-2 89.00% | vlm |
| 06 | `projects/06_qwen25vl_plantvillage` | Qwen2.5-VL-7B + LoRA, production app | Apr 2026 | **98.50%** | vlm |
| 07 | `projects/07_plantdx_plantvillage` | PlantDx: Qwen2.5-VL-7B + LoRA, 15 classes | May 2026 | 98.00% | vlm |
| 08 | `projects/08_plantvlm_expert_captions` | Qwen2.5-VL-7B + LoRA on 150 expert captions | Aug 2026 | 2 of 6 (exploratory) | vlm |
| 09 | `projects/09_plantdx_mlx_knowledge_grounded` | Knowledge-grounded Qwen2.5-VL QLoRA on **Apple MLX** (tomato, mango) | Jul 2026 | 93.7% / 82.0% | Mac only |
| 10 | `projects/10_tih_tomato_qwen_qlora` | Qwen2.5-VL-7B 4-bit QLoRA, TIH 11-class tomato (**ICAIN 2026 paper**) | Aug – Sep 2026 | **92.76% test, 69.32% cross-domain** | tih |

VLM results (04–08) are all on the same 200 PlantVillage validation images, so they compare directly.
Numbers 05 and 11 moved to `failed_experiments/` (Florence-2 and the SigLIP prototype).

## What did not work (`failed_experiments/`)

Florence-2 fine-tuning (never finished, 0% zero-shot), LLaVA-1.6 (2%), LLaVA-1.5 (collation error), the BLIP-2
June 0% scoring bug, the first Qwen run and its fixes, zero-shot VLMs (8.5%), PlantVLM on 30 images (2 of 6), an
aborted TIH training run, the untrained SigLIP/OpenCLIP prototype, and EfficientNet in the Keras SoyNet run. Each
has its code (where any survived), what happened and the lesson: start at `failed_experiments/README.md`.

## Everything from the fellowship report, version by version

`docs/REPORT_CHECKLIST.md` takes every entry of the Chanakya report (month-wise timeline, December tasks, papers,
every model, versions v0.1–v14.0, mango, soybean v1.0–v3.0, appendices) and points to its code, model, evaluation
report and run command here, or says plainly why something is not available.

- **Older versions** of a project are kept as they were in `projects/<project>/versions/` (CottonCrop 4–6 Jan,
  the 27 Jan integrated pipeline, the 13 Apr crop app, the 15 May PlantDx).
- **Every commit, branch and tag** of our seven GitHub repositories: `history/README.md`, with offline git
  bundles (e.g. tag `v1.0.0` = the July MLX release, branch `feat/tomato-dataset-migration` = v11.0).
- **Per-epoch checkpoints** of the TIH runs and PlantVLM, older model copies and the failed EfficientNet files:
  `python setup\download_models.py --all` (+2.6 GB). The ~64 GB of step-by-step checkpoints that stay on the
  workstation are listed in `docs/CHECKPOINT_INVENTORY.md`.
- **Month by month:** `docs/monthly/` (December 2025 – October 2026).
- **Reports:** `docs/reports/` (fellowship report, both versions; PoCRA reports v1 and v2; PlantDx history;
  presentations).

## Repository structure

```
handover/
├── README.md                      # this page
├── run_app.py                     # menu: start any Streamlit app with the right environment
├── demo_app.py                    # Streamlit: every model on one page
├── setup/
│   ├── setup_envs.py              # creates .venvs\keras, soynet, vlm, florence, tih from frozen requirements
│   ├── download_models.py         # pulls every trained model (checksummed) from github.com/iAakash1/TIH-IoT
│   ├── check_setup.py             # READY / MISSING check for envs, models, datasets, GPU
│   ├── models_manifest.json       # which model file goes where (+ size, SHA-256)
│   └── requirements/              # exact package versions per environment
├── projects/                      # one folder per working model, in the order we did them
│   ├── 01_cnn_keras_tomato/       # predict.py, evaluate.py, preprocessing/, validation/, model/
│   ├── 02_cnn_pytorch_convnext/   # predict.py, evaluate.py, notebooks/, splits/
│   ├── 03_soybean_soynet/         # predict_soynet.py, Keras + PyTorch pipelines, results/
│   ├── 04_llava_blip2/            # blip2_march/ (March code), blip2_rerun_june/, outputs/ (adapters)
│   ├── 06_qwen25vl_plantvillage/  # qwen/ (April run), crop_disease_app/ (Streamlit app)
│   ├── 07_plantdx_plantvillage/   # PlantDx package: core/, utils/, scripts/, app/
│   ├── 08_plantvlm_expert_captions/  # plantvlm/ package, scripts/, runs/
│   ├── 09_plantdx_mlx_knowledge_grounded/  # MLX: results, configs, Mac commands (code is in 10's src/plantdx)
│   └── 10_tih_tomato_qwen_qlora/  # plantdx package (src/), streamlit_app.py, configs/, checkpoints/, reports/
├── failed_experiments/            # everything that did not work out, with code and lessons
├── evaluation/                    # vlm_eval.py, vlm_predict.py + every October 2026 evaluation report
├── history/                       # every commit/branch/tag of our repositories + offline git bundles
└── docs/
    ├── REPORT_CHECKLIST.md        # every entry of the fellowship report -> where it is here
    ├── HANDOVER_REPORT.md         # model-wise report: data, method, results, failures
    ├── HOW_TO_RUN.md              # 2-page guide: run, retrain, edit, common errors
    ├── datasets.md                # every dataset, where it is, exact splits
    ├── publication.md             # ICAIN 2026 paper, project reports, papers studied
    ├── publication/               # ICAIN 2026 camera-ready paper (PDF)
    ├── monthly/                   # one page per month, Dec 2025 - Oct 2026
    ├── CHECKPOINT_INVENTORY.md    # training checkpoints kept on the workstation
    ├── timeline_verification.md   # how each date was checked against commits and files
    ├── vlm_plantvillage_comparison.md  # all PlantVillage VLMs side by side
    └── reports/                   # fellowship report (both versions), PoCRA v1 and v2, presentations
```

## Good to know

- **Datasets are not in git** (size, licences; the TIH tomato data is private). They are on the TIH workstation;
  paths are in `docs/datasets.md` and can be changed with environment variables.
- **Models are not in git either**: `setup/download_models.py` fetches them (3.6 GB). They also stay on the
  TIH workstation in this folder.
- **MLX (project 09)** ran on an Apple Silicon Mac. The adapters stayed on the Mac, but all the code is here: it is
  the same `plantdx` package as project 10 (`src/plantdx/training/mlx_runner.py`), also published at
  github.com/iAakash1/experimentation. Results, configs and Mac commands are in the 09 README.
- **Paper:** `docs/publication/ICAIN2026_paper953_camera_ready.pdf` (accepted 1 Oct 2026, registered 21 Oct 2026).
