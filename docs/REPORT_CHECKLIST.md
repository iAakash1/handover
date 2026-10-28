# Every entry of the fellowship report → where it is in this repository

Source: `docs/reports/Kamaaksha_Chanakya_Report_TIH_v2.0.docx` (latest version; the 21 October copy is next to it).
Each row says where the code, model and evaluation report of that entry are, and how to run it. ✅ = here and
working, 📄 = documented here (nothing runnable exists), ⛔ = not available anywhere we can reach (reason given).

## 1.2 Month-wise timeline

| Month | Report entry | In this repository |
|---|---|---|
| Dec 2025 | Image operations with OpenCV/PIL vs NumPy scratch (Week 1) | 📄 tasks listed below; the Week-1 day notebooks are not on this workstation or in our GitHub repositories |
| Dec 2025 | Training preprocessing, quality checks, augmentation, inference guardrails (Week 2) | ✅ `projects/01_cnn_keras_tomato/preprocessing/` (and the original package in `versions/v0.2-v0.3_cottoncrop_2026-01-06/preprocessing/`) |
| Dec 2025 | CNN theory and CNN training on PlantVillage (Week 3) | ✅ `projects/01_cnn_keras_tomato/model/train_cnn.py` |
| Dec 2025 | Custom CNN trained from scratch (TF/Keras), 98.51% | ✅ model `projects/01_cnn_keras_tomato/model/best_model_cnn.keras`, report `evaluation/keras/custom_cnn_*`, run `predict.py --model cnn` |
| Jan 2026 | EfficientNetB3, MobileNetV2, ConvNeXt-Tiny (Keras) with quality gate and crop validation | ✅ `projects/01_cnn_keras_tomato` (`predict.py --model efficientnet / mobilenet / convnext --quality-check`), reports `evaluation/keras/` |
| Jan 2026 | ConvNeXt-Tiny (PyTorch, RTX 4500 Ada), 99.92% | ✅ `projects/02_cnn_pytorch_convnext`, report `evaluation/convnext_torch/` |
| Jan 2026 | Crop-validation embeddings and integrated pipeline | ✅ `projects/01_cnn_keras_tomato/validation/`, `generate_reference_embeddings.py`; original `versions/v0.5_integrated_pipeline_2026-01-27` |
| Jan 2026 | SoyNet soybean training (Keras and PyTorch runs), ResNet50 95.16% | ✅ `projects/03_soybean_soynet` (v1.0–v3.0 table in its README, `predict_soynet.py`) |
| Jan 2026 | Soybean training on the VDI (TIH-provided pipeline and dataset) | ⛔ the model stayed on the VDI, which is currently unavailable (noted in `projects/03_soybean_soynet/README.md`) |
| Feb 2026 | Video editing for the PoCRA project; app testing | 📄 no code or model; the videos were delivered to PoCRA and are not in our folders |
| Feb 2026 | Attendance at the AI for Agri conference | 📄 `docs/monthly/2026-02_February.md` |
| Mar 2026 | Literature study of VLMs (8 papers) | ✅ `docs/publication.md` (titles, authors, links) |
| Mar 2026 | LLaVA-1.6-Mistral-7B LoRA, 2.00% | ✅ adapter `projects/04_llava_blip2/outputs/llava_plantvillage/best`, training script and story `failed_experiments/02_llava16_mistral_lora`, report `evaluation/llava_v16/` |
| Mar 2026 | BLIP-2 OPT-2.7B LoRA, 89.00% | ✅ code `projects/04_llava_blip2/blip2_march/`, adapter `outputs/blip2_plantvillage/best`, report `evaluation/blip2_march/`, app `python run_app.py 7` |
| Apr 2026 | Research papers read (four downloaded) | ✅ `docs/publication.md` papers 1–4 |
| Apr 2026 | VLM code repository | ✅ Crop_Disease_VLM: every commit as a folder, see `history/README.md` |
| Apr 2026 | Florence-2 scripts | ✅ `failed_experiments/01_florence2_finetuning`, report `evaluation/florence_zeroshot/` |
| Apr 2026 | Qwen2.5-VL-7B LoRA fine-tuned, 98.50% | ✅ `projects/06_qwen25vl_plantvillage/qwen/`, report `qwen/evaluation_report.json`, app `python run_app.py 5` |
| Apr 2026 | CLIP OOD gate, parser, validator, GradCAM pipeline | ✅ first version `projects/06_qwen25vl_plantvillage/versions/v5.0_crop_app_2026-04-13`, final `crop_disease_app/` (`python run_app.py 2`) |
| May 2026 | PlantDx restructured pipeline (15 classes), run_001 | ✅ first version `projects/07_plantdx_plantvillage/versions/v6.0_first_restructure_2026-05-15`; runs in `projects/07_plantdx_plantvillage/outputs/runs/` |
| May 2026 | Evaluation and Streamlit inference, 98.00% | ✅ `projects/07_plantdx_plantvillage` (report `outputs/evaluations/report.json`, app `python run_app.py 4`) |
| Jun 2026 | BLIP-2 1-epoch re-run with evaluation | ✅ `projects/04_llava_blip2/blip2_rerun_june`, report `evaluation/blip2_june/`, the 0% story `failed_experiments/04_blip2_june_zero_score` |
| Jun 2026 | Florence-2 re-attempt | ✅ `failed_experiments/01_florence2_finetuning` (`quick_train_florence2.py`) |
| Jun 2026 | Expert-validated caption image set | ✅ `projects/08_plantvlm_expert_captions/1stJuly` (captions in git, 30 images on the workstation) |
| Jun 2026 | Seven-VLM zero-shot benchmark | 📄 run on the Mac with MLX-VLM; results in `docs/reports/06-PoCRA-PCCOE-Tomato-Mango-Report-Version1.docx`, summary `failed_experiments/06_zero_shot_baselines` (no script survived) |
| Jul 2026 | 150 expert-revised captions | ✅ `projects/08_plantvlm_expert_captions/data/` and `1stJuly/` |
| Jul 2026 | Knowledge-grounded MLX pipeline, tomato 7.3% → 93.7%, mango 17.5% → 82.0% | ✅ code `projects/10_tih_tomato_qwen_qlora/src/plantdx` (exact July state: tag `v1.0.0` in `history/experimentation.bundle`), results and configs `projects/09_plantdx_mlx_knowledge_grounded`; ⛔ adapters stayed on the Mac |
| Jul 2026 | Streamlit demo; v1.0.0 | ✅ `projects/10_tih_tomato_qwen_qlora/streamlit_app.py`; tag `v1.0.0` in the bundle |
| Aug 2026 | PlantVLM on 150 expert-validated captions, 2/6 | ✅ `projects/08_plantvlm_expert_captions` (best adapter + epochs 1–5 with `--all`), `failed_experiments/07_plantvlm_expert_captions` |
| Aug 2026 | TIH Tomato_12thAug26 dataset received | ✅ described in `docs/datasets.md` (private, not redistributed) |
| Aug 2026 | PlantDx migrated (v11.0) | ✅ branch `feat/tomato-dataset-migration` (`cab7194`, `ac6a7f8`) in `history/experimentation.bundle` and `history/Tomato_12th_Aug_2026.bundle` |
| Aug 2026 | SigLIP/OpenCLIP prototype | ✅ `failed_experiments/09_siglip_openclip_prototype` |
| Sep 2026 | QLoRA smoke run (11 classes) | ✅ `projects/10_tih_tomato_qwen_qlora/reports/tih_tomato_qwen_qlora/` and `checkpoints/tih_tomato_qwen_qlora/run_config.json`; the aborted automatic run `failed_experiments/08_tih_aborted_training_run` |
| Sep 2026 | 'strong' run, 87.98% | ✅ `checkpoints/tih_tomato_qwen_qlora_strong` (+ checkpoint-656/1312/1968 with `--all`), `reports/tih_tomato_qwen_qlora_strong/` |
| Sep 2026 | 'moreaccurate' run, 92.76% | ✅ `checkpoints/tih_tomato_moreaccurate` (+ checkpoint-656/1312/1908 with `--all`), `reports/tih_tomato_moreaccurate/`, app `python run_app.py 1` |
| Sep 2026 | Cross-domain benchmark, 176 images, 69.32% | ✅ `reports/tih_tomato_moreaccurate_external/`, run `company evaluate-external` |
| Sep 2026 | Repository clean-up for release | ✅ branch `refactor/professional-research-layout` (`66a40bc`) = `projects/10_tih_tomato_qwen_qlora` |
| Oct 2026 | ICAIN 2026 paper accepted (1 Oct) | ✅ `docs/publication/ICAIN2026_paper953_camera_ready.pdf`, `docs/publication.md` |
| Oct 2026 | Registration and fee payment | 📄 `docs/publication.md` |
| Oct 2026 | All code, models and reports consolidated; this report | ✅ this repository; reports in `docs/reports/`, re-evaluations in `evaluation/` |

## 1.3 December 2025 foundation tasks

| Task | In this repository |
|---|---|
| Week 1, Day 1–4: read/display/save, metadata, BGR–RGB, resize/crop/rotate/flip, interpolation, blurs, colour spaces, sharpening, Sobel, Canny, scratch NumPy versions and timing | 📄 the same operations are used in `projects/01_cnn_keras_tomato/preprocessing/` (`filters.py`, `color_processing.py`); the day-wise notebooks themselves are not on this workstation |
| Week 2, Day 1: `preprocess_for_training()` | ✅ `projects/01_cnn_keras_tomato/preprocessing/preprocess_training.py` |
| Week 2, Day 2: quality checks (blur > 100, brightness 40–220, contrast > 20, noise) and augmentation | ✅ `preprocessing/quality_checks.py`, `preprocessing/augment_training.py` |
| Week 2, Day 3: training-time vs inference-time preprocessing | ✅ `preprocessing/preprocess_training.py` vs `preprocessing/preprocess_inference.py` |
| Week 2, Day 4: `preprocess_for_inference()` with all checks (min 128 px) | ✅ `preprocessing/preprocess_inference.py` |
| Week 2, Day 5: one integrated module | ✅ `preprocessing/__init__.py` |
| Week 3, Day 1–3, 5: CNN theory, activations, losses, documentation | 📄 study tasks, no code |
| Week 3, Day 4: CNN from scratch in TensorFlow and PyTorch | ✅ TensorFlow `model/train_cnn.py`; PyTorch comparison carried into `projects/02_cnn_pytorch_convnext` |

## 1.4 – 1.5 Papers and publication

All 8 papers: `docs/publication.md`. ICAIN 2026 paper (ID 953), camera-ready PDF: `docs/publication/`.

## 2 – 3 Crops, models and versions

### Tomato: models used

| Model in the report | In this repository |
|---|---|
| Custom CNN from scratch (Keras) | ✅ project 01 |
| EfficientNetB3, MobileNetV2, ConvNeXt-Tiny (Keras) | ✅ project 01 |
| ConvNeXt-Tiny (PyTorch) | ✅ project 02 |
| LLaVA 1.5 Vicuna-7B (no checkpoint retained) | ⛔ `failed_experiments/03_llava15` (no code or weights survived) |
| LLaVA 1.6 Mistral-7B, LoRA | ✅ adapter in project 04, script in `failed_experiments/02_llava16_mistral_lora` |
| BLIP-2 OPT-2.7B, LoRA | ✅ project 04 (March and June) |
| Florence-2-base-ft | ✅ `failed_experiments/01_florence2_finetuning` |
| Qwen2.5-VL-7B LoRA r32, structured captions | ✅ projects 06 and 07 |
| Qwen2.5-VL-7B-4bit QLoRA r16 (MLX), knowledge-grounded | ✅ code + results (project 09 / 10); ⛔ adapters on the Mac |
| Qwen2.5-VL-7B LoRA r16, 150 expert captions (PlantVLM) | ✅ project 08 |
| Qwen2.5-VL-7B 4-bit NF4 QLoRA r32 and r64, TIH 11 classes | ✅ project 10 (strong = r32, moreaccurate = r64) |
| Seven-VLM zero-shot benchmark (Qwen2.5-VL, Qwen3-VL, Gemma-3, InternVL3, LLaMA-3.2-Vision, Molmo, DeepSeek-VL2) | 📄 PoCRA report; `failed_experiments/06_zero_shot_baselines` |
| CLIP ViT-B/32 OOD screening | ✅ `projects/06_qwen25vl_plantvillage/crop_disease_app/utils/` |

### Tomato: experiment versions

| Version | Date | Code | Model | Report |
|---|---|---|---|---|
| v0.1 Custom CNN | 29 Dec 2025 | `projects/01_cnn_keras_tomato/model/train_cnn.py` | `model/best_model_cnn.keras` | `evaluation/keras/custom_cnn_best_tomato_test` |
| v0.2 EfficientNetB3 + gates | 04 Jan 2026 | `projects/01_cnn_keras_tomato/versions/v0.2_cottoncrop_2026-01-04` | `model/best_model_efficientNetB3.keras` | `evaluation/keras/efficientnetb3_keras` |
| v0.3 MobileNetV2, ConvNeXt (Keras) | 06 Jan 2026 | same folder | `model/best_model_mobilenet.keras`, `best_model_convnext.keras` | `evaluation/keras/` |
| v0.4 ConvNeXt (PyTorch) | 08 Jan 2026 | `projects/02_cnn_pytorch_convnext` | `models/convnext_tomato_best.pth` | `evaluation/convnext_torch` |
| v0.5 Integrated pipeline | 21–27 Jan 2026 | `projects/01_cnn_keras_tomato/versions/v0.5_integrated_pipeline_2026-01-27` | same models (byte-identical) | – |
| v1.0 LLaVA 1.6 | 24–25 Mar 2026 | `failed_experiments/02_llava16_mistral_lora` | `projects/04_llava_blip2/outputs/llava_plantvillage/best` | `evaluation/llava_v16` |
| v2.0 BLIP-2 | 25 Mar 2026 | `projects/04_llava_blip2/blip2_march` | `outputs/blip2_plantvillage/best` | `evaluation/blip2_march` |
| v3.0 Florence-2 | 03 Apr / 15 Jun 2026 | `failed_experiments/01_florence2_finetuning` | none (did not finish) | `evaluation/florence_zeroshot` |
| v4.0 Qwen2.5-VL LoRA | 06–07 Apr 2026 | `projects/06_qwen25vl_plantvillage/qwen` (fixes: `failed_experiments/05_qwen_april_first_attempt`) | `qwen/outputs/qwen25vl_plantvillage/best` | `qwen/evaluation_report.json` |
| v5.0 CLIP gate, parser, validator, GradCAM | 13 Apr 2026 | `projects/06_qwen25vl_plantvillage/versions/v5.0_crop_app_2026-04-13`, `crop_disease_app` | same adapter | – |
| v6.0 PlantDx | 15–18 May 2026 | `projects/07_plantdx_plantvillage` (+ `versions/v6.0_first_restructure_2026-05-15`) | `outputs/checkpoints/best` | `outputs/evaluations/report.json` |
| v7.0 BLIP-2 re-run | 15 Jun 2026 | `projects/04_llava_blip2/blip2_rerun_june` | `blip2_rerun_june/outputs/blip2_plantvillage/best` | `evaluation/blip2_june` |
| v8.0 Seven-VLM zero-shot | 25 Jun 2026 | – (Mac) | – | PoCRA report in `docs/reports/` |
| v9.0 Knowledge-grounded MLX | 11–17 Jul 2026 | `projects/10_tih_tomato_qwen_qlora/src/plantdx`, tag `v1.0.0` | on the Mac | `projects/09_plantdx_mlx_knowledge_grounded/results/` |
| v10.0 PlantVLM | 05 Aug 2026 | `projects/08_plantvlm_expert_captions` | `runs/run_002_fold0/checkpoints/best` (+ epochs) | `runs/run_002_fold0/evaluation/val_report.json` |
| v11.0 PlantDx migrated to TIH data | 12 Aug 2026 | branch `feat/tomato-dataset-migration` in `history/` | – | – |
| v12.0 TIH 'strong' | 15 Sep 2026 | `projects/10_tih_tomato_qwen_qlora`, config `configs/train/qwen25vl_tih_tomato_strong.yaml` | `checkpoints/tih_tomato_qwen_qlora_strong` | `reports/tih_tomato_qwen_qlora_strong/test_finetuned` |
| v13.0 TIH 'moreaccurate' | 16 Sep 2026 | config `configs/train/qwen25vl_tih_tomato_moreaccurate.yaml` | `checkpoints/tih_tomato_moreaccurate` | `reports/tih_tomato_moreaccurate/test_finetuned` |
| v14.0 External evaluation | 17 Sep 2026 | `src/plantdx/company/external_evaluate.py` | same | `reports/tih_tomato_moreaccurate_external` |

### Mango

| Version | In this repository |
|---|---|
| Mango baseline (zero-shot 17.5%, macro-F1 0.128) | ✅ `projects/09_plantdx_mlx_knowledge_grounded/results/zero_shot/` |
| Mango QLoRA (82.0%, macro-F1 0.811; commits `380b18e`, `673ed77`) | ✅ config `projects/09_plantdx_mlx_knowledge_grounded/configs/train/qwen25vl_mango.yaml`, results `results/mlx_pipeline/`, report `evaluation/mlx_from_confusion_matrices`; code at those commits in `history/experimentation.bundle`; ⛔ adapter on the Mac |

### Soybean

| Version | In this repository |
|---|---|
| v1.0 First ResNet50 (Keras, `main.py`) | ✅ `projects/03_soybean_soynet/saved_models/soynet_resnet50_best.keras` |
| v2.0 Keras multi-backbone (ResNet50 95.16%, MobileNetV2 89.07%) | ✅ `projects/03_soybean_soynet/results/experiments/*/soynet_*_best.keras`, `saved_models/*_finetuned.keras`; the EfficientNet files that never built: `failed_experiments/10_soynet_keras_efficientnet` |
| v3.0 PyTorch multi-backbone (best EfficientNetB3 93.93%) | ✅ `projects/03_soybean_soynet/results/experiments/*/model.pth`, `predict_soynet.py` |
| Model trained on the VDI with the TIH dataset | ⛔ on the VDI (currently unavailable) |

## Appendix A – C

| Report item | In this repository |
|---|---|
| A: split files | `projects/02_cnn_pytorch_convnext/splits/split_70_15_15.csv`, `projects/07_plantdx_plantvillage/data/splits/`, `projects/10_tih_tomato_qwen_qlora/checkpoints/tih_tomato_moreaccurate/dataset_manifest.json`, `projects/03_soybean_soynet/splits/` |
| A: model paths | every model is in `setup/models_manifest.json` (`python setup/download_models.py`, older versions with `--all`) |
| B: the seven GitHub repositories, all branches and commits | `history/README.md` (+ offline bundles) |
| C: TIH-IoT month folders | `docs/monthly/` (one page per month) and the projects listed above |
| Intermediate training checkpoints | `docs/CHECKPOINT_INVENTORY.md` |

## Everything else found on the workstation

| What | Where |
|---|---|
| Unedited originals of every file that was changed for the handover (paths, headers, links) | `archive/original_files/` |
| Earlier copies from the Downloads folder: Florence-2 patch and quick-train (15 Jun), BLIP-2 quick-train (12 Jun), crop app config/README (13 Apr), first Qwen app fix `fix_app.py` (7 Apr) | `failed_experiments/01_florence2_finetuning/earlier_copies_2026-06-15`, `projects/04_llava_blip2/blip2_rerun_june/earlier_copies_2026-06-12`, `projects/06_qwen25vl_plantvillage/versions/v5.0_crop_app_2026-04-13/earlier_copies`, `failed_experiments/05_qwen_april_first_attempt/fix_app.py` |
| Knowledge base v2 for the TIH tomato migration (12 Aug) | `projects/10_tih_tomato_qwen_qlora/private_archive/knowledge_base/dkb_v2_tomato_only_high_depth_rectified.json` |
| VLM presentation (31 Mar) | `docs/reports/vlm_presentation_2026-03-31.html` |
| First Qwen diagnosis output (8 Apr) | `projects/06_qwen25vl_plantvillage/qwen/sample_diagnosis_2026-04-08.txt` |
| Photos uploaded to the PlantDx app on 18 May | `projects/07_plantdx_plantvillage/data/raw/uploads/` |
| Field test photos (31 Mar, 9 Apr) | `projects/06_qwen25vl_plantvillage/test_images/` |
| CottonCrop at every commit (4 and 6 Jan) | `projects/01_cnn_keras_tomato/versions/` |
| Datasets (PlantVillage, SoyNet zip and raw photos, TIH tomato, MangoLeafBD) | not redistributed; locations in `docs/datasets.md` |
