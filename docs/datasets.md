# Datasets used

None of the images are in git (size, licences, and the TIH data is private). On the TIH workstation they are
already in the folders below, and every script defaults to these paths. On another machine, copy the data
anywhere and set the environment variable in the first column.

| Variable | Default on the TIH workstation | What it is | Used by |
|---|---|---|---|
| `PLANTVILLAGE_DIR` | `C:\Users\Admin\Desktop\VLM\PlantVillage` | PlantVillage, 15 class folders (tomato, potato, pepper), ~20,600 images | 04, 05, 06, 07, evaluation |
| `TOMATO_DATASET_DIR` | `C:\Users\Admin\Desktop\iitb_fellowship\data\raw\PlantVillage` | Same PlantVillage folders (Dec–Jan copy) | 01 |
| `TOMATO_SPLIT_DIR` | `C:\Users\Admin\Desktop\iitb_fellowship\data` | 10 tomato classes split 70/15/15 into `train/ val/ test/` (11,203 / 2,397 / 2,411) | 01, 02 |
| `SOYNET_DIR` | `C:\Users\Admin\Desktop\latest\SoyNet\Preprocessing_SoyNet_Data\Camera Clicks_256_256` | SoyNet camera photos, 256×256: 3,164 diseased + 556 healthy | 03 |
| *(inside project 08)* | `projects/08_plantvlm_expert_captions/1stJuly/Dataset` | 30 PlantVillage tomato images (6 diseases × 5) with 150 expert-revised captions | 08 |
| `--dataset-root` | `C:\Users\Admin\Desktop\TIH_IITB_August_2026\Tomato_12thAug26\Dataset` | **TIH private tomato dataset**: 11 classes, 7,095 images (6,545 after de-duplication); `testing/` = 176 images from another source for the cross-domain benchmark | 10, 11 |
| – | not on this machine | MangoLeafBD: 8 classes, 4,000 images (public, CC BY 4.0, https://data.mendeley.com/datasets/hxsnvwty3r/1) | 09 (MLX, Mac only) |
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
