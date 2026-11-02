# Handover: crop disease detection with CNNs and Vision-Language Models

TIH IIT Bombay · Chanakya Fellowship (Agri-AI 2025) · December 2025 – September 2026  
Team: Aakash Trimbak Jawle, Kamaaksha Rajendra Kharul · Faculty mentor: Dr. Santwana Gudadhe (PCCOE Pune)  
TIH mentors: Harish Lokawad, Hrishikesh Patil, Rutuja Chavan

Everything we built, trained and measured during the fellowship: code, trained models, evaluation reports,
datasets used, our ICAIN 2026 paper, and what did **not** work.

**Start with [`docs/README_HANDOVER.md`](docs/README_HANDOVER.md)** (where everything is) and
**[`docs/HANDOVER_REPORT.md`](docs/HANDOVER_REPORT.md)** (the model-wise handover report). Every model has a
one-to-two-page `HANDOVER_GUIDE.md` in its project folder.

## Start here (Windows PowerShell, from this folder)

```powershell
python setup\setup_envs.py          # 1. creates the 5 Python environments in .venvs\ (Python 3.10, ~25 GB, once)
python setup\download_models.py     # 2. downloads all trained models (~4.0 GB) into the right folders
python setup\check_setup.py         # 3. must end with READY (lists anything missing and how to fix it)
python run_app.py                   # 4. pick an app from the menu; it opens in the browser
```

On the TIH workstation the datasets are already in place. Elsewhere, see [`docs/datasets.md`](docs/datasets.md). To check a
clone without installing anything: `python setup\verify_repository.py --with-projects` (links, configs, Python
syntax, model manifest, and the MLX project's own checks).

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
ever run downloads the base model from Hugging Face (Qwen2.5-VL 16 GB, BLIP-2 14 GB). The MLX models (project 09)
have their own demo on the Mac.

## What is here, model by model

| # | Folder and guide | Model | When | Best result | Env |
|---|---|---|---|---|---|
| 00 | [`projects/00_tih_foundation_weeks_dec2025`](projects/00_tih_foundation_weeks_dec2025/README.md) | TIH Week 1–2 notebooks: image operations (OpenCV vs NumPy), preprocessing, quality checks | Dec 2025 | – | any Python |
| 01 | [`projects/01_cnn_keras_tomato`](projects/01_cnn_keras_tomato/README.md) · [guide](projects/01_cnn_keras_tomato/HANDOVER_GUIDE.md) | Custom CNN, MobileNetV2, EfficientNetB3, ConvNeXt-Tiny (Keras) | Dec 2025 – Jan 2026 | ConvNeXt 96.15% (val) | keras |
| 02 | [`projects/02_cnn_pytorch_convnext`](projects/02_cnn_pytorch_convnext/README.md) · [guide](projects/02_cnn_pytorch_convnext/HANDOVER_GUIDE.md) | ConvNeXt-Tiny (PyTorch) | Jan 2026 | **99.92% test** | vlm |
| 03 | [`projects/03_soybean_soynet`](projects/03_soybean_soynet/README.md) · [guide](projects/03_soybean_soynet/HANDOVER_GUIDE.md) | ResNet50, EfficientNetB0/B3, MobileNetV2 (soybean) | Jan 2026 | ResNet50 95.16% | soynet |
| 04 | [`projects/04_llava_blip2`](projects/04_llava_blip2/README.md) · [guide](projects/04_llava_blip2/HANDOVER_GUIDE.md) | BLIP-2 + LoRA (March and June), LLaVA-1.6 adapter | Mar, Jun 2026 | BLIP-2 89.00% | vlm |
| 06 | [`projects/06_qwen25vl_plantvillage`](projects/06_qwen25vl_plantvillage/README.md) · [guide](projects/06_qwen25vl_plantvillage/HANDOVER_GUIDE.md) | Qwen2.5-VL-7B + LoRA, production app | Apr 2026 | **98.50%** | vlm |
| 07 | [`projects/07_plantdx_plantvillage`](projects/07_plantdx_plantvillage/README.md) · [guide](projects/07_plantdx_plantvillage/HANDOVER_GUIDE.md) | PlantDx: Qwen2.5-VL-7B + LoRA, 15 classes | May 2026 | 98.00% | vlm |
| 08 | [`projects/08_plantvlm_expert_captions`](projects/08_plantvlm_expert_captions/README.md) · [guide](projects/08_plantvlm_expert_captions/HANDOVER_GUIDE.md) | Qwen2.5-VL-7B + LoRA on 150 expert captions | Aug 2026 | 2 of 6 (exploratory) | vlm |
| 09 | [`projects/09_plantdx_mlx_knowledge_grounded`](projects/09_plantdx_mlx_knowledge_grounded/README.md) · [guide](projects/09_plantdx_mlx_knowledge_grounded/HANDOVER_GUIDE.md) | Knowledge-grounded Qwen2.5-VL QLoRA on **Apple MLX** (tomato, mango) | Jul 2026 (retrained Nov 2026) | see below | Mac only |
| 10 | [`projects/10_tih_tomato_qwen_qlora`](projects/10_tih_tomato_qwen_qlora/README.md) · [guide](projects/10_tih_tomato_qwen_qlora/HANDOVER_GUIDE.md) | Qwen2.5-VL-7B 4-bit QLoRA, TIH 11-class tomato (**ICAIN 2026 paper**) | Aug – Sep 2026 | **92.76% test, 69.32% cross-domain** | tih |

VLM results of 04, 06 and 07 are on the same 200 PlantVillage validation images, so they compare directly.
Numbers 05 and 11 (Florence-2, the SigLIP prototype) are in [`failed_experiments/`](failed_experiments/README.md).

### MLX models (Apple Silicon, project 09)

| Model | Architecture | Data (test images) | Test accuracy, macro-F1 | Served weights |
|---|---|---|---|---|
| [`qwen25vl_tomato_qlora`](projects/09_plantdx_mlx_knowledge_grounded/models/qwen25vl_tomato_qlora/README.md) | Qwen2.5-VL-7B-Instruct 4-bit + QLoRA r16 (`mlx_vlm.lora`) | PlantVillage tomato, 10 classes (910) | base 7.3%; July adapter 93.7%, 0.919; **Nov adapter 90.9%, 0.877** | Nov 2026, iteration 13,200 |
| [`qwen25vl_mango_qlora`](projects/09_plantdx_mlx_knowledge_grounded/models/qwen25vl_mango_qlora/README.md) | same | MangoLeafBD, 8 classes (200) | base 17.5%; July adapter 82.0%, 0.811; **Nov adapter 86.0%, 0.891** | Nov 2026, final iteration |

The July adapter files were no longer on the Mac; both models were retrained with the same code, data and command
([`retrained_2026-11/`](projects/09_plantdx_mlx_knowledge_grounded/retrained_2026-11/README.md)), and `setup/download_models.py` fetches the November weights.
Requirements: macOS on Apple Silicon (trained on an M4 Pro, 24 GB), Python 3.12, `mlx` 0.31.2, `mlx-vlm` 0.6.4
([`requirements-mlx.txt`](projects/09_plantdx_mlx_knowledge_grounded/requirements-mlx.txt)). Reproduction:
[`docs/REPRODUCTION.md`](projects/09_plantdx_mlx_knowledge_grounded/docs/REPRODUCTION.md); what is in git, downloaded or generated:
[`docs/ARTIFACTS.md`](projects/09_plantdx_mlx_knowledge_grounded/docs/ARTIFACTS.md); presenting it:
[`PRESENTATION_GUIDE.md`](projects/09_plantdx_mlx_knowledge_grounded/PRESENTATION_GUIDE.md).

## What did not work ([`failed_experiments/`](failed_experiments/README.md))

Florence-2 fine-tuning (never finished, 0% zero-shot), LLaVA-1.6 (2%), LLaVA-1.5 (collation error), the BLIP-2
June 0% scoring issue, the first Qwen run and its fixes, zero-shot VLMs (8.5%), PlantVLM on 30 images (2 of 6), an
aborted TIH training run, the untrained SigLIP/OpenCLIP prototype, EfficientNet in the Keras SoyNet run, and the
earlier Mac approaches (PlantCaption, caption sets, 4- and 7-VLM MLX zero-shot benchmarks). Each has its code
(where any survived), what happened and the lesson.

## Everything from the fellowship report, version by version

[`docs/REPORT_CHECKLIST.md`](docs/REPORT_CHECKLIST.md) takes every entry of the Chanakya report (month-wise timeline, December tasks, papers,
every model, versions v0.1–v14.0, mango, soybean v1.0–v3.0, appendices) and points to its code, model, evaluation
report and run command here, or says plainly why something is not available.

- **Older versions** of projects 01, 04, 06 and 07 are kept as they were in [`archive/previous_versions/`](archive/previous_versions/)
  (CottonCrop 4–6 Jan, the 27 Jan integrated pipeline, the 13 Apr crop app, the 15 May PlantDx).
- **Every commit, branch and tag** of our seven GitHub repositories: [`history/README.md`](history/README.md), with offline git
  bundles (e.g. tag `v1.0.0` = the July MLX release, branch `feat/tomato-dataset-migration` = v11.0).
- **Per-epoch checkpoints** of the TIH runs and PlantVLM, older model copies and the failed EfficientNet files:
  `python setup\download_models.py --all` (+3.6 GB). The ~64 GB of step-by-step checkpoints that stay on the
  workstation are listed in [`docs/CHECKPOINT_INVENTORY.md`](docs/CHECKPOINT_INVENTORY.md).
- **Originals** of every file that was changed for the handover: [`archive/original_files/`](archive/README.md).
- **Month by month:** [`docs/monthly/`](docs/monthly/) (December 2025 – October 2026).
- **Reports:** [`docs/reports/`](docs/reports/) (fellowship report, both versions; PoCRA reports v1 and v2; PlantDx history;
  completion report; presentations).

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
│   ├── verify_repository.py       # links, JSON/YAML, Python syntax, manifest (+ tests/), no installs needed
│   ├── models_manifest.json       # which model file goes where (+ size, SHA-256)
│   └── requirements/              # exact package versions per environment
├── projects/                      # the final implementation of each model; each has README.md + HANDOVER_GUIDE.md
│   ├── 00_tih_foundation_weeks_dec2025/  # Week 1–2 day notebooks (image ops, preprocessing, quality checks)
│   ├── 01_cnn_keras_tomato/       # predict.py, evaluate.py, preprocessing/, validation/, model/
│   ├── 02_cnn_pytorch_convnext/   # predict.py, evaluate.py, notebooks/, splits/
│   ├── 03_soybean_soynet/         # predict_soynet.py, Keras + PyTorch pipelines, results/
│   ├── 04_llava_blip2/            # blip2_march/ (March code), blip2_rerun_june/, outputs/ (adapters)
│   ├── 06_qwen25vl_plantvillage/  # qwen/ (April run), crop_disease_app/ (Streamlit app)
│   ├── 07_plantdx_plantvillage/   # PlantDx package: core/, utils/, scripts/, app/
│   ├── 08_plantvlm_expert_captions/  # plantvlm/ package, scripts/, runs/
│   ├── 09_plantdx_mlx_knowledge_grounded/  # MLX (Mac): plantdx package, tomato + mango images, reports, logs, demo app,
│   │                              #   models/ (one card per model), retrained_2026-11/, docs/REPRODUCTION.md, docs/ARTIFACTS.md
│   └── 10_tih_tomato_qwen_qlora/  # plantdx package (src/), streamlit_app.py, configs/, checkpoints/, reports/
├── failed_experiments/            # what did not work and earlier approaches, with code and lessons
├── evaluation/                    # vlm_eval.py, vlm_predict.py + every October 2026 evaluation report
├── history/                       # every commit/branch/tag of our repositories + offline git bundles
├── archive/                       # previous_versions/ of the final projects; original_files/ edited for the handover
└── docs/
    ├── README_HANDOVER.md         # where to start, where everything is
    ├── HANDOVER_REPORT.md         # the model-wise handover report
    ├── HOW_TO_RUN.md              # setup, run, retrain, edit, common errors
    ├── datasets.md                # dataset register: sources, access, counts, splits, paths
    ├── publication.md             # papers and their status, project reports, papers studied
    ├── publication/               # ICAIN 2026 camera-ready paper; drafts/ (PlantDx MLX paper, OOD paper, ICAIN submission)
    ├── REPORT_CHECKLIST.md        # every entry of the fellowship report -> where it is here
    ├── monthly/                   # one page per month, Dec 2025 - Oct 2026
    ├── CHECKPOINT_INVENTORY.md    # training checkpoints kept on the workstation
    ├── timeline_verification.md   # each date and the commit or file it comes from
    ├── vlm_plantvillage_comparison.md  # all PlantVillage VLMs side by side
    └── reports/                   # fellowship reports, completion report, PoCRA v1 and v2, proposal and review decks, presentations
```

## Good to know

- **Datasets are not in git** (size, licences; the TIH tomato data is private and is accessed through TIH-IoT). They
  are on the TIH workstation; paths are in [`docs/datasets.md`](docs/datasets.md) and can be changed with environment variables. The
  exception is the MLX project 09: the public PlantVillage tomato and MangoLeafBD images it was trained on are in
  its `tomato/` and `mango/`.
- **Models are not in git either**: `setup/download_models.py` fetches them (4.0 GB) from github.com/iAakash1/TIH-IoT.
  They also stay on the TIH workstation in this folder.
- **CUDA vs MLX:** projects 02–08 and 10 run on the NVIDIA workstation; project 09 runs only on an Apple Silicon Mac.
- **Paper:** [`docs/publication/ICAIN2026_paper953_camera_ready.pdf`](docs/publication/ICAIN2026_paper953_camera_ready.pdf) (ICAIN 2026, accepted 1 Oct 2026,
  registered 21 Oct 2026).
