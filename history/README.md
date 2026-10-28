# Git history of every repository we used

Every commit, branch and tag of our code repositories, so any version mentioned in the reports can be checked
out exactly as it was. The small repositories are included as git bundles (complete offline copies); restore one
with plain git, no GitHub access needed:

```powershell
git clone history\experimentation.bundle experimentation          # all branches and tags
cd experimentation
git checkout v1.0.0                                                # e.g. the July MLX release
git checkout feat/tomato-dataset-migration                         # e.g. PlantDx moved to the TIH data (12 Aug)
```

## Report version → where its exact code is

| Report version | Date | Commit / tag | Also in this repository as files |
|---|---|---|---|
| Tomato v0.2–v0.3, Keras transfer learning | 04–06 Jan 2026 | CottonCrop `5cc1f6b` | `projects/01_cnn_keras_tomato/versions/v0.2-v0.3_cottoncrop_2026-01-06` |
| Tomato v0.5, integrated pipeline | 21–27 Jan 2026 | (no git; folder copy) | `projects/01_cnn_keras_tomato/versions/v0.5_integrated_pipeline_2026-01-27` |
| Tomato v2.0, BLIP-2 March code | 01 Apr 2026 | Crop_Disease_VLM `44e9b67` (local history) | `projects/04_llava_blip2/blip2_march` |
| Tomato v3.0 / v4.0, Florence-2 and Qwen scripts | 09 Apr 2026 | Crop_Disease_VLM `5dcff36` (local history) | `failed_experiments/01_florence2_finetuning`, `projects/06_qwen25vl_plantvillage/qwen`, `qwen_variant` |
| Tomato v5.0, CLIP gate + parser + GradCAM | 13 Apr 2026 | Crop_Disease_VLM `6b4e667` | `projects/06_qwen25vl_plantvillage/versions/v5.0_crop_app_2026-04-13` |
| Crop disease app, final | 13 Jul 2026 | Crop_Disease_VLM `6a8760b` | `projects/06_qwen25vl_plantvillage/crop_disease_app` |
| Tomato v6.0, PlantDx first restructure | 15 May 2026 | (folder copy) | `projects/07_plantdx_plantvillage/versions/v6.0_first_restructure_2026-05-15` |
| PlantDx published | 02–03 Jun 2026 | kharulkamaaksha/VLM `f07c678`, iAakash1/VLM PR #1 | `projects/07_plantdx_plantvillage` |
| Tomato v9.0, knowledge-grounded MLX | 11–17 Jul 2026 | experimentation `v1.0.0` (`66d428e`), `9f08d99` | `projects/09_plantdx_mlx_knowledge_grounded` (results, configs) |
| Mango baseline + QLoRA | 14 Jul 2026 | experimentation `380b18e`, `673ed77` (`v0.2.0`) | `projects/09_plantdx_mlx_knowledge_grounded` |
| Tomato v11.0, PlantDx migrated to TIH data | 12 Aug 2026 | `cab7194`, `ac6a7f8` (branch `feat/tomato-dataset-migration`) | in the bundles |
| Tomato v12.0, TIH 'strong' | 15 Sep 2026 | `8eb2cd6` | `projects/10_tih_tomato_qwen_qlora` (run `tih_tomato_qwen_qlora_strong`) |
| Tomato v13.0, TIH 'moreaccurate' | 16 Sep 2026 | `791702a` | `projects/10_tih_tomato_qwen_qlora` (run `tih_tomato_moreaccurate`) |
| Tomato v14.0, external evaluation + release layout | 17 Sep 2026 | `66a40bc` (branch `refactor/professional-research-layout`) | `projects/10_tih_tomato_qwen_qlora` |

Not bundled: Crop_Disease_VLM (its local history is 339 MB because early commits contain images; every one of its
four commits is in this repository as a folder, see the table), CottonCrop (367 MB with data; its code is the v0.2–v0.3
folder) and TIH-IoT (the model store; github.com/iAakash1/TIH-IoT).

## kharulkamaaksha/experimentation (nested copy on the Mac)

`~/Desktop/experiments/experimentation` on the Mac was a second copy of the PlantDx MLX project, pushed to
github.com/kharulkamaaksha/experimentation, which no longer exists. Its files are identical to
`projects/09_plantdx_mlx_knowledge_grounded`; its only commit is kept offline: `history/kharulkamaaksha_experimentation.bundle`.

| Date | Commit | Message | Branches containing it |
|---|---|---|---|
| 2026-07-30 | `d4eb1aa` | Initial commit | main |

## iAakash1/experimentation

https://github.com/iAakash1/experimentation · full history offline: `history/experimentation.bundle`

Branches: `feat/mango-training` `feat/tih-tomato-qwen` `main` `refactor/professional-research-layout` 

Tags: v0.1-scaffold v0.1.0-ci-green v0.2.0 v0.3-normalization v0.4-ontology v0.5-stabilization v0.5-vocabulary v0.6-corpus-freeze v0.6-generation v0.8-tomato-training v1.0-evaluation v1.0.0 

| Date | Commit | Message | Branches containing it |
|---|---|---|---|
| 2026-07-11 | `12efddd` | feat: scaffold PlantDx repository (tag: v0.1-scaffold ) | feat/mango-training feat/tih-tomato-qwen main refactor/professional-research-layout  |
| 2026-07-11 | `3e73d0f` | feat: implement dataset audit and normalization engine (tag: v0.3-normalization ) | feat/mango-training feat/tih-tomato-qwen main refactor/professional-research-layout  |
| 2026-07-12 | `2125fc9` | feat: implement deterministic ontology compiler (tag: v0.4-ontology ) | feat/mango-training feat/tih-tomato-qwen main refactor/professional-research-layout  |
| 2026-07-12 | `584f49e` | chore: stabilize repository and finalize developer documentation (tag: v0.5-stabilization ) | feat/mango-training feat/tih-tomato-qwen main refactor/professional-research-layout  |
| 2026-07-12 | `7b45bb2` | feat: implement vocabulary builder and symptom lexicon (tag: v0.5-vocabulary ) | feat/mango-training feat/tih-tomato-qwen main refactor/professional-research-layout  |
| 2026-07-13 | `05a98e1` | feat: implement deterministic caption generation pipeline (tag: v0.6-corpus-freeze v0.6-generation ) | feat/mango-training feat/tih-tomato-qwen main refactor/professional-research-layout  |
| 2026-07-13 | `8ad2726` | feat: complete tomato Qwen2.5-VL LoRA training pipeline (tag: v0.8-tomato-training ) | feat/mango-training feat/tih-tomato-qwen main refactor/professional-research-layout  |
| 2026-07-14 | `2d11adf` | feat: implement end-to-end VLM evaluation framework (tag: v1.0-evaluation ) | feat/mango-training feat/tih-tomato-qwen main refactor/professional-research-layout  |
| 2026-07-14 | `ba457c4` | feat: implement end-to-end VLM evaluation framework | feat/mango-training feat/tih-tomato-qwen main refactor/professional-research-layout  |
| 2026-07-14 | `638b7bd` | fix: stabilize evaluation dependencies and CI | feat/mango-training feat/tih-tomato-qwen main refactor/professional-research-layout  |
| 2026-07-14 | `3774492` | test: make evaluation fixtures self-contained (tag: v0.1.0-ci-green ) | feat/mango-training feat/tih-tomato-qwen main refactor/professional-research-layout  |
| 2026-07-14 | `380b18e` | feat: add mango training pipeline | feat/mango-training feat/tih-tomato-qwen main refactor/professional-research-layout  |
| 2026-07-14 | `673ed77` | feat: generalize evaluation pipeline for multiple crops (tag: v0.2.0 ) | feat/mango-training feat/tih-tomato-qwen main refactor/professional-research-layout  |
| 2026-07-17 | `088c134` | feat(app): add Streamlit inference/demo app for the trained adapters | feat/tih-tomato-qwen main refactor/professional-research-layout  |
| 2026-07-17 | `66d428e` | docs: overhaul README as the public entry point with results + figures (tag: v1.0.0 ) | feat/tih-tomato-qwen main refactor/professional-research-layout  |
| 2026-07-17 | `9f08d99` | docs: polish repository presentation for public release | feat/tih-tomato-qwen main refactor/professional-research-layout  |
| 2026-08-12 | `cab7194` | feat: migrate PlantDx to tomato dataset | feat/tih-tomato-qwen refactor/professional-research-layout  |
| 2026-08-12 | `ac6a7f8` | chore: remove temporary DKB files | feat/tih-tomato-qwen refactor/professional-research-layout  |
| 2026-09-15 | `8eb2cd6` | Add TIH tomato Qwen2.5-VL fine-tuning workflow | feat/tih-tomato-qwen refactor/professional-research-layout  |
| 2026-09-16 | `791702a` | Add TIH tomato moreaccurate Qwen2.5-VL experiment | feat/tih-tomato-qwen refactor/professional-research-layout  |
| 2026-09-17 | `78740fc` | Separate MLX research and TIH benchmark experiments | refactor/professional-research-layout  |
| 2026-09-17 | `ff3ab2b` | Organize benchmark results and experiment artifacts | refactor/professional-research-layout  |
| 2026-09-17 | `66a40bc` | Clean repository documentation and public release structure | refactor/professional-research-layout  |

## kharulkamaaksha/Tomato_12th_Aug_2026

https://github.com/kharulkamaaksha/Tomato_12th_Aug_2026 · full history offline: `history/Tomato_12th_Aug_2026.bundle`

Branches: `feat/tih-tomato-qwen` `feat/tomato-dataset-migration` 

| Date | Commit | Message | Branches containing it |
|---|---|---|---|
| 2026-07-11 | `12efddd` | feat: scaffold PlantDx repository | feat/tih-tomato-qwen feat/tomato-dataset-migration  |
| 2026-07-11 | `3e73d0f` | feat: implement dataset audit and normalization engine | feat/tih-tomato-qwen feat/tomato-dataset-migration  |
| 2026-07-12 | `2125fc9` | feat: implement deterministic ontology compiler | feat/tih-tomato-qwen feat/tomato-dataset-migration  |
| 2026-07-12 | `584f49e` | chore: stabilize repository and finalize developer documentation | feat/tih-tomato-qwen feat/tomato-dataset-migration  |
| 2026-07-12 | `7b45bb2` | feat: implement vocabulary builder and symptom lexicon | feat/tih-tomato-qwen feat/tomato-dataset-migration  |
| 2026-07-13 | `05a98e1` | feat: implement deterministic caption generation pipeline | feat/tih-tomato-qwen feat/tomato-dataset-migration  |
| 2026-07-13 | `8ad2726` | feat: complete tomato Qwen2.5-VL LoRA training pipeline | feat/tih-tomato-qwen feat/tomato-dataset-migration  |
| 2026-07-14 | `2d11adf` | feat: implement end-to-end VLM evaluation framework | feat/tih-tomato-qwen feat/tomato-dataset-migration  |
| 2026-07-14 | `ba457c4` | feat: implement end-to-end VLM evaluation framework | feat/tih-tomato-qwen feat/tomato-dataset-migration  |
| 2026-07-14 | `638b7bd` | fix: stabilize evaluation dependencies and CI | feat/tih-tomato-qwen feat/tomato-dataset-migration  |
| 2026-07-14 | `3774492` | test: make evaluation fixtures self-contained | feat/tih-tomato-qwen feat/tomato-dataset-migration  |
| 2026-07-14 | `380b18e` | feat: add mango training pipeline | feat/tih-tomato-qwen feat/tomato-dataset-migration  |
| 2026-07-14 | `673ed77` | feat: generalize evaluation pipeline for multiple crops | feat/tih-tomato-qwen feat/tomato-dataset-migration  |
| 2026-07-17 | `088c134` | feat(app): add Streamlit inference/demo app for the trained adapters | feat/tih-tomato-qwen feat/tomato-dataset-migration  |
| 2026-07-17 | `66d428e` | docs: overhaul README as the public entry point with results + figures | feat/tih-tomato-qwen feat/tomato-dataset-migration  |
| 2026-07-17 | `9f08d99` | docs: polish repository presentation for public release | feat/tih-tomato-qwen feat/tomato-dataset-migration  |
| 2026-08-12 | `cab7194` | feat: migrate PlantDx to tomato dataset | feat/tih-tomato-qwen feat/tomato-dataset-migration  |
| 2026-08-12 | `ac6a7f8` | chore: remove temporary DKB files | feat/tih-tomato-qwen feat/tomato-dataset-migration  |
| 2026-09-15 | `8eb2cd6` | Add TIH tomato Qwen2.5-VL fine-tuning workflow | feat/tih-tomato-qwen  |

## kharulkamaaksha/Crop_Disease_VLM

https://github.com/kharulkamaaksha/Crop_Disease_VLM

Branches: `main` 

| Date | Commit | Message | Branches containing it |
|---|---|---|---|
| 2026-04-01 | `44e9b670` | Initial commit (code only) | main  |
| 2026-04-09 | `5dcff364` | Initial commit: VLM crop disease model (BLIP2 + Qwen pipeline) | main  |
| 2026-04-13 | `6b4e6670` | Crop Disease VLM - production pipeline |  |
| 2026-07-13 | `6a8760b2` | Polish: cross-platform config, requirements.txt, .gitignore, setup checker, README overhaul |  |

## kharulkamaaksha/VLM

https://github.com/kharulkamaaksha/VLM · full history offline: `history/kharulkamaaksha_VLM.bundle`

Branches: `docs-improvements` `main` 

| Date | Commit | Message | Branches containing it |
|---|---|---|---|
| 2026-06-02 | `f07c678` | Initial commit | docs-improvements main  |
| 2026-06-03 | `6a914c4` | Improve README structure and setup instructions | docs-improvements  |

## iAakash1/VLM

https://github.com/iAakash1/VLM · full history offline: `history/iAakash1_VLM.bundle`

Branches: `main` 

| Date | Commit | Message | Branches containing it |
|---|---|---|---|
| 2026-06-02 | `f07c678` | Initial commit | main  |
| 2026-06-03 | `6a914c4` | Improve README structure and setup instructions | main  |
| 2026-06-03 | `a594401` | Merge pull request #1 from kharulkamaaksha/docs-improvements | main  |

## iAakash1/CottonCrop

https://github.com/iAakash1/CottonCrop

Branches: `main` 

| Date | Commit | Message | Branches containing it |
|---|---|---|---|
| 2026-01-04 | `7f4f63cb` | Initial commit | main  |
| 2026-01-04 | `7214dbd2` | Add gitignore to exclude venv | main  |
| 2026-01-04 | `77b13fff` | Add complete CottonCrop project | main  |
| 2026-01-06 | `5cc1f6be` | Add MobileNet and ConvNeXt models with trained weights (Git LFS) | main  |

## iAakash1/TIH-IoT

https://github.com/iAakash1/TIH-IoT

Branches: `main` 

| Date | Commit | Message | Branches containing it |
|---|---|---|---|
| 2026-10-21 | `5edd920` | Add all fellowship code, models, results and documents (Dec 2025 - Sep 2026) | main  |
| 2026-10-21 | `4c6c10f` | Add READMEs, October 2026 evaluation reports, split files and timeline verification | main  |
| 2026-10-21 | `fe1e412` | Add month-wise folders (Dec 2025 - Sep 2026) with findings and evaluation reports | main  |
| 2026-10-21 | `e9231a6` | Correct README facts, add June BLIP-2 re-evaluation, record ICAIN registration (21 Oct 2026) | main  |
| 2026-10-21 | `5cc6de0` | Reorganise repository month-wise (December 2025 - September 2026) | main  |
| 2026-10-28 | `4cf3cdc` | Add per-epoch training checkpoints for the TIH tomato runs and PlantVLM | main  |

