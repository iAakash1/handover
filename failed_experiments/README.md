# Failed and abandoned experiments

Not everything we tried worked. This folder keeps those attempts together, each with its code (where any
survived), what happened, and what we learned, so nobody has to repeat them. The working models are in
`projects/`.

| # | Folder | Month | What we tried | What happened | Lesson |
|---|---|---|---|---|---|
| 01 | `01_florence2_finetuning` | Apr, Jun 2026 | Fine-tune Florence-2-base-ft on PlantVillage captions | Training never finished: five incompatibilities between its remote code and transformers 5.3. Zero-shot (transformers 4.46.3): **0%**, scene captions only | Pin the transformers version a model was written for |
| 02 | `02_llava16_mistral_lora` | Mar 2026 | LLaVA-1.6-Mistral-7B + LoRA r64, 1,700 steps (~17 h) | **2.00%** accuracy (macro-F1 0.0151); no disease named in 178 of 200 answers | Long, generic answers; BLIP-2 (89%) and later Qwen2.5-VL (98.5%) did far better |
| 03 | `03_llava15` | Mar 2026 | LLaVA-1.5-7B + LoRA | Stopped by a batch-collation error before any checkpoint; no code or weights survived | Images with different tile counts can't be stacked into one batch: train with batch size 1 |
| 04 | `04_blip2_june_zero_score` | Jun 2026 | 1-epoch BLIP-2 re-run | Its own June evaluation reported **0%**; scored with the April protocol it is **89.00%** | The scoring script, not the model, was wrong: always re-check a surprising 0 |
| 05 | `05_qwen_april_first_attempt` | Apr 2026 | First Qwen2.5-VL fine-tuning run | Crashed until four things were fixed: answers cut at 256 tokens, processor truncation breaking image tokens, a collate function that did not pad, Flash-Attention 2 / wrong model class / CLIP loading in the app | The `fix_*.py` scripts here are the patches; the run after them gave 98.50% |
| 06 | `06_zero_shot_baselines` | Apr – Sep 2026 | Using VLMs without fine-tuning | Qwen2.5-VL-7B: **8.50%** (PlantVillage), **8.47%** (TIH tomato); Florence-2: 0%; June seven-VLM benchmark: unreliable, Qwen3-VL-8B degenerate on 16 of 30 images | General VLMs don't know plant diseases; fine-tuning is required |
| 07 | `07_plantvlm_expert_captions` | Aug 2026 | Qwen2.5-VL + LoRA on 150 expert-written captions of 30 images | Right disease for **2 of 6** validation images (chance: 1 of 6) | Good captions can't make up for too few images |
| 08 | `08_tih_aborted_training_run` | Sep 2026 | A full TIH training campaign started automatically from the CLI | Stopped at step ~40 of 1,968, no adapter written; kept for its zero-shot baseline (8.47%) | Official runs start only from the Streamlit **RUN TRAINING** button |
| 09 | `09_siglip_openclip_prototype` | Aug 2026 | Contrastive SigLIP / OpenCLIP classifier for the TIH data | Written, never trained: Qwen2.5-VL QLoRA was chosen (92.76%) | Kept as a lighter option for the future |
| 10 | `10_soynet_keras_efficientnet` | Jan 2026 | EfficientNetB0/B3 in the Keras SoyNet run | The models did not build; only empty 6 KB files were saved (removed) | The PyTorch run trained both (93.93%, 93.39%) |

Every number above comes from a report in this repository (`evaluation/`, the project folders, or the
folders here).
