# Handover: crop disease detection with CNNs and Vision-Language Models

TIH IIT Bombay · Chanakya Fellowship (Agri-AI 2025) · December 2025 – September 2026  
Team: Aakash Trimbak Jawle, Kamaaksha Rajendra Kharul · Faculty mentor: Dr. Santwana Gudadhe (PCCOE Pune)  
TIH mentors: Harish Lokawad, Hrishikesh Patil, Rutuja Chavan

Everything we built, trained and measured during the fellowship: code, trained models, evaluation reports,
datasets used, papers and how to run each model. Every model folder has its own README with exact commands.

## Start here (4 steps, Windows PowerShell, from this folder)

```powershell
python setup\setup_envs.py          # 1. creates the 5 Python environments in .venvs\ (Python 3.10, ~25 GB, once)
python setup\download_models.py     # 2. downloads all trained models (~3.6 GB) into the right folders
python setup\check_setup.py         # 3. must end with READY (lists anything missing and how to fix it)
cd projects\01_cnn_keras_tomato     # 4. try a model
..\..\.venvs\keras\Scripts\python predict.py --image sample.jpg
```

On the TIH workstation the datasets are already in place. Elsewhere, see `docs/datasets.md`.

## What is here, model by model

| # | Folder | Model | When | Best result | Env |
|---|---|---|---|---|---|
| 01 | `projects/01_cnn_keras_tomato` | Custom CNN, MobileNetV2, EfficientNetB3, ConvNeXt-Tiny (Keras) | Dec 2025 – Jan 2026 | ConvNeXt 96.15% (val) | keras |
| 02 | `projects/02_cnn_pytorch_convnext` | ConvNeXt-Tiny (PyTorch) | Jan 2026 | **99.92% test** | vlm |
| 03 | `projects/03_soybean_soynet` | ResNet50, EfficientNetB0/B3, MobileNetV2 (soybean) | Jan 2026 | ResNet50 95.16% | soynet |
| 04 | `projects/04_llava_blip2` | LLaVA-1.6 + LoRA, BLIP-2 + LoRA | Mar, Jun 2026 | BLIP-2 89.00% | vlm |
| 05 | `projects/05_florence2` | Florence-2 (fine-tuning failed; zero-shot) | Apr, Jun 2026 | 0% (dropped) | florence |
| 06 | `projects/06_qwen25vl_plantvillage` | Qwen2.5-VL-7B + LoRA, production app | Apr 2026 | **98.50%** | vlm |
| 07 | `projects/07_plantdx_plantvillage` | PlantDx: Qwen2.5-VL-7B + LoRA, 15 classes | May 2026 | 98.00% | vlm |
| 08 | `projects/08_plantvlm_expert_captions` | Qwen2.5-VL-7B + LoRA on 150 expert captions | Aug 2026 | 2 of 6 (exploratory) | vlm |
| 09 | `projects/09_plantdx_mlx_knowledge_grounded` | Knowledge-grounded Qwen2.5-VL QLoRA on **Apple MLX** (tomato, mango) | Jul 2026 | 93.7% / 82.0% | Mac only |
| 10 | `projects/10_tih_tomato_qwen_qlora` | Qwen2.5-VL-7B 4-bit QLoRA, TIH 11-class tomato (**ICAIN 2026 paper**) | Aug – Sep 2026 | **92.76% test, 69.32% cross-domain** | tih |
| 11 | `projects/11_siglip_openclip_prototype` | SigLIP / OpenCLIP prototype (code only, not trained) | Aug 2026 | – | – |

VLM results (04–08) are all on the same 200 PlantVillage validation images, so they compare directly.

## Repository structure

```
handover/
├── README.md                      # this page
├── setup/
│   ├── setup_envs.py              # creates .venvs\keras, soynet, vlm, florence, tih from frozen requirements
│   ├── download_models.py         # pulls every trained model (checksummed) from github.com/iAakash1/TIH-IoT
│   ├── check_setup.py             # READY / MISSING check for envs, models, datasets, GPU
│   ├── models_manifest.json       # which model file goes where (+ size, SHA-256)
│   └── requirements/              # exact package versions per environment
├── projects/                      # one folder per model / experiment, in the order we did them
│   ├── 01_cnn_keras_tomato/       # predict.py, evaluate.py, preprocessing/, validation/, model/
│   ├── 02_cnn_pytorch_convnext/   # predict.py, evaluate.py, notebooks/, splits/
│   ├── 03_soybean_soynet/         # predict_soynet.py, Keras + PyTorch pipelines, results/
│   ├── 04_llava_blip2/            # March adapters, June BLIP-2 pipeline
│   ├── 05_florence2/              # Florence-2 scripts (kept as written)
│   ├── 06_qwen25vl_plantvillage/  # qwen/ (April run), crop_disease_app/ (Streamlit app)
│   ├── 07_plantdx_plantvillage/   # PlantDx package: core/, utils/, scripts/, app/
│   ├── 08_plantvlm_expert_captions/  # plantvlm/ package, scripts/, runs/
│   ├── 09_plantdx_mlx_knowledge_grounded/  # MLX section: results, configs, how to run on a Mac
│   ├── 10_tih_tomato_qwen_qlora/  # plantdx package (src/), configs/, checkpoints/, reports/, tests/
│   └── 11_siglip_openclip_prototype/
├── evaluation/                    # vlm_eval.py, vlm_predict.py + every October 2026 evaluation report
└── docs/
    ├── HANDOVER_REPORT.md         # model-wise report: data, method, results, how to run
    ├── HOW_TO_RUN.md              # 2-page guide: run, retrain, edit, common errors
    ├── datasets.md                # every dataset, where it is, exact splits
    ├── publication.md             # ICAIN 2026 paper, project reports, papers studied
    └── reports/                   # PoCRA and fellowship reports, presentations
```

## Good to know

- **Datasets are not in git** (size, licences; the TIH tomato data is private). They are on the TIH workstation;
  paths are in `docs/datasets.md` and can be changed with environment variables.
- **Models are not in git either** — `setup/download_models.py` fetches them (3.6 GB). They also stay on the
  TIH workstation in this folder.
- **MLX models (project 09)** were trained on an Apple Silicon Mac and cannot run here; their results, configs
  and the commands to reproduce them on a Mac are in that folder.
- The first run of a VLM downloads its base model from Hugging Face (Qwen2.5-VL 16 GB, LLaVA 15 GB, BLIP-2 14 GB).
