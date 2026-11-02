# Datasets used

## Register

| Dataset | Source and access | Classes | Images | Preprocessing and checks | Splits (exact counts) | Used by |
|---|---|---|---|---|---|---|
| PlantVillage (tomato, potato, pepper) | public (Hughes & Salathé 2015, github.com/spMohanty/PlantVillage-Dataset); not in git | 15 folders | ~20,600 | resized per model; captions from per-class templates for the VLMs | VLMs 90/10, seed 42 (project 06: 18,574 / 2,064 caption records); shared 200-image evaluation set | 04, 06, 07, evaluation |
| PlantVillage tomato (Dec–Jan copy) | public; not in git | 10 | 16,011 in the split | 224×224 (32×32 custom CNN), Week-2 quality checks at inference | 70/15/15 per class: 11,203 / 2,397 / 2,411 (`splits/split_70_15_15.csv`); Keras runs: `validation_split` 0.2 (3,198 validation) | 01, 02 |
| PlantVillage tomato (MLX copy) | public; **in git** (`projects/09_…/tomato/raw/PlantVillage/`) | 10 | 18,160 (18,006 `*.JPG` used) | `plantdx audit` (class distribution, duplicates, corrupt images; `projects/09_…/reports/`), `plantdx normalize` (checksum `b9556420…`), 448×448 | image-grouped stratified, seed 20260711: 16,201 / 895 / 910 | 09 |
| MangoLeafBD | public, CC BY 4.0 (Ahmed et al. 2023, data.mendeley.com/datasets/hxsnvwty3r/1); **in git** (`projects/09_…/mango/raw/MangoLeafBD/`) | 8 | 4,000 (500 per class) | as above (checksum `6fa29822…`) | seed 20260711: 3,600 / 200 / 200 | 09 |
| SoyNet camera photos | provided for the task; not in git | 2 (diseased, healthy) | 3,720 (3,164 + 556) | resized to 256×256 | stratified 70/15/15, seed 42: 2,603 / 557 / 560 (`projects/03_…/splits/`) | 03 |
| TIH `Tomato_12thAug26` | **private**, TIH-IoT; only on the TIH workstation, access through the TIH-IoT mentors; not in git | 11 | 7,095 → 6,545 | integrity checks, SHA-256 de-duplication, removal of images present under two classes; 532 px longest side | stratified 80/10/10, seed 20260911: 5,247 / 649 / 649 (`projects/10_…/checkpoints/tih_tomato_moreaccurate/dataset_manifest.json`); external set `Dataset/testing`: 176 images, 10 classes | 10 |
| Expert-captioned tomato leaves | 30 PlantVillage images with 150 expert-revised captions; images not in git, captions in `projects/08_…/1stJuly/Revised_captions.xlsx` | 6 | 30 | – | grouped, 3 / 1 / 1 images per class (fold 0) | 08, 7-VLM benchmark |

## Locations and environment variables

PlantVillage, SoyNet and the TIH data are not in git (size, licences, and the TIH data is private). On the TIH
workstation they are already in the folders below, and every script defaults to these paths. On another machine,
copy the data anywhere and set the environment variable in the first column.

| Variable | Default on the TIH workstation | What it is | Used by |
|---|---|---|---|
| `PLANTVILLAGE_DIR` | `C:\Users\Admin\Desktop\VLM\PlantVillage` | PlantVillage, 15 class folders (tomato, potato, pepper), ~20,600 images | 04, 05, 06, 07, evaluation |
| `TOMATO_DATASET_DIR` | `C:\Users\Admin\Desktop\iitb_fellowship\data\raw\PlantVillage` | Same PlantVillage folders (Dec–Jan copy) | 01 |
| `TOMATO_SPLIT_DIR` | `C:\Users\Admin\Desktop\iitb_fellowship\data` | 10 tomato classes split 70/15/15 into `train/ val/ test/` (11,203 / 2,397 / 2,411) | 01, 02 |
| `SOYNET_DIR` | `C:\Users\Admin\Desktop\latest\SoyNet\Preprocessing_SoyNet_Data\Camera Clicks_256_256` | SoyNet camera photos, 256×256: 3,164 diseased + 556 healthy | 03 |
| *(inside project 08)* | `projects/08_plantvlm_expert_captions/1stJuly/Dataset` | 30 PlantVillage tomato images (6 diseases × 5) with 150 expert-revised captions | 08 |
| `--dataset-root` | `C:\Users\Admin\Desktop\TIH_IITB_August_2026\Tomato_12thAug26\Dataset` | **TIH private tomato dataset**: 11 classes, 7,095 images (6,545 after de-duplication); `testing/` = 176 images from another source for the cross-domain benchmark | 10, 11 |
| *(inside project 09)* | `projects/09_plantdx_mlx_knowledge_grounded/mango/raw/MangoLeafBD` | MangoLeafBD: 8 classes, 4,000 images (public, CC BY 4.0, https://data.mendeley.com/datasets/hxsnvwty3r/1) | 09 (MLX, Mac only) |
| *(inside project 09)* | `projects/09_plantdx_mlx_knowledge_grounded/tomato/raw/PlantVillage` | PlantVillage tomato, 10 classes, `train/` + `val/`, 18,160 images (the copy the MLX runs used) | 09 (MLX, Mac only) |
| – | `C:/Users/Admin/Desktop/latest/SoyNet/Raw_SoyNet_Data` | SoyNet raw photos before resizing: `Camera Clicks` (1,194 images) and `Mobile Click` (169 images; our training config used only the camera photos) | 03 |
| – | on the TIH VDI | TIH-provided soybean dataset used for the VDI soybean model (VDI currently unavailable) | 03 |

Set a variable for the current PowerShell window, then run the project command:

```powershell
$env:PLANTVILLAGE_DIR = "E:\data\PlantVillage"
```

## Exact splits (so results can be reproduced)

| Project | Split | File |
|---|---|---|
| 02 ConvNeXt | 70/15/15 per class | `projects/02_cnn_pytorch_convnext/splits/split_70_15_15.csv` |
| 03 SoyNet | 70/15/15 stratified, seed 42 | `projects/03_soybean_soynet/splits/soynet_pytorch_split_70_15_15_seed42.csv` |
| 06 Qwen (April) | 90/10, seed 42 | `projects/06_qwen25vl_plantvillage/qwen/captions.json` (`train` / `val`) |
| 07 PlantDx | 90/10, seed 42 | `projects/07_plantdx_plantvillage/data/splits/splits.json` |
| 08 PlantVLM | 3/1/1 images per class, fold 0 | `projects/08_plantvlm_expert_captions/data/{train,val,test}.jsonl` |
| 10 TIH tomato | 80/10/10 stratified, seed 20260911 (5,247 / 649 / 649) | `projects/10_tih_tomato_qwen_qlora/checkpoints/tih_tomato_moreaccurate/dataset_manifest.json` |

The caption files store the original absolute image paths (`C:\Users\Admin\Desktop\VLM\PlantVillage\...`).
`evaluation/vlm_eval.py` swaps that prefix for `PLANTVILLAGE_DIR` automatically.
