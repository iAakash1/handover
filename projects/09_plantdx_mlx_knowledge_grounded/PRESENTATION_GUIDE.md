# Presentation guide: PlantDx on Apple Silicon (MLX)

How to walk someone through this work: what was done, in what order, which file shows each step, the results,
and answers to the questions that usually come up. Every number here comes from the files linked next to it.

## 1. Where to open

| What | Link |
|---|---|
| Overview: results, graphs, timeline, team, run commands | [`README.md`](README.md) |
| Zero-shot experiments | [`earlier_experiments/README.md`](earlier_experiments/README.md) |
| Paper draft on this work | [`PlantDx_Knowledge_Grounded_Instruction_Tuning_draft_2026-08-21.pdf`](../../docs/publication/drafts/PlantDx_Knowledge_Grounded_Instruction_Tuning_draft_2026-08-21.pdf) |

## 2. The 30-second version

> We first tested ready-made vision-language models on leaf photos with no training (zero-shot). They mostly
> could not name the disease: the best reached 23%. So we did not let an AI write our training captions. We built a
> cited Disease Knowledge Base of 18 diseases and compiled 1,966 captions from it with deterministic code. Then we
> fine-tuned Qwen2.5-VL-7B on a Mac with Apple's MLX framework, using QLoRA, which trains only 0.49% of the
> parameters. Tomato went from 7.3% to 93.7% accuracy, and mango from 17.5% to 82.0%.

## 3. What was done, step by step, and the file that shows it

| Step | Date (2026) | What to say | Open |
|---|---|---|---|
| 1. Zero-shot benchmark, 4 models | 25 Jun | Qwen2.5-VL, Gemma-3, InternVL3 and Llama-3.2-Vision on 16 leaves, one fixed prompt. Exact disease match only 0–6% per model. | [`benchmark.py`](earlier_experiments/03_zero_shot_4vlm_benchmark_2026-06-25/scripts/benchmark.py), [`final_report.md`](earlier_experiments/03_zero_shot_4vlm_benchmark_2026-06-25/report/final_report.md) |
| 2. Zero-shot benchmark, 7 models | 1 Jul | 30 tomato leaves (the 30 expert-captioned images of project 08). Best model InternVL3-8B at 23%; Qwen3-VL-8B gave unusable output on 16 of 30. | [`run_benchmark.py`](earlier_experiments/04_zero_shot_7vlm_benchmark_2026-07-01/scripts/run_benchmark.py), [`model_runner.py`](earlier_experiments/04_zero_shot_7vlm_benchmark_2026-07-01/scripts/model_runner.py), [`PlantDx_VLM_Evaluation_Report.md`](earlier_experiments/04_zero_shot_7vlm_benchmark_2026-07-01/report/PlantDx_VLM_Evaluation_Report.md) |
| 3. Knowledge base | 11 Jul | 18 diseases (10 tomato, 8 mango): symptoms, look-alike diseases, allowed and forbidden words, references. | [`knowledge_base/dkb.json`](knowledge_base/dkb.json), [`DKB_report.md`](knowledge_base/DKB_report.md) |
| 4. Data audit and cleaning | 12–13 Jul | Duplicate and corrupt-image checks, then each crop normalized into one class folder per disease. | [`src/plantdx/audit/`](src/plantdx/audit/), [`src/plantdx/normalization/`](src/plantdx/normalization/), [`reports/class_distribution.csv`](reports/class_distribution.csv) |
| 5. Knowledge base → captions | 12–13 Jul | Ontology → controlled vocabulary → concept models → templates → 1,966 captions. No AI model in this path. | [`ontology/`](src/plantdx/ontology/), [`vocabulary/`](src/plantdx/vocabulary/), [`concepts/`](src/plantdx/concepts/), [`generation/`](src/plantdx/generation/), [`corpus/`](src/plantdx/corpus/); output [`artifacts/corpus/captions.json`](artifacts/corpus/captions.json) |
| 6. Training data | 13–14 Jul | Each image paired with a question and a caption; split with a fixed seed (20260711). | [`src/plantdx/dataset/splits.py`](src/plantdx/dataset/splits.py), [`train.jsonl`](artifacts/training/qwen25vl_tomato_qlora/dataset/train.jsonl) |
| 7. Fine-tuning on MLX | 13–14 Jul | QLoRA on Qwen2.5-VL-7B (4-bit), tomato first, then mango. | [`configs/train/qwen25vl_tomato.yaml`](configs/train/qwen25vl_tomato.yaml), [`mlx_runner.py`](src/plantdx/training/mlx_runner.py), [`train.log`](logs/qwen25vl_tomato_qlora/train.log) |
| 8. Evaluation | 14 Jul | Base vs fine-tuned on held-out test images: accuracy, F1, BLEU, ROUGE, METEOR, CIDEr, BERTScore, hallucination checks. | [`src/plantdx/evaluation/`](src/plantdx/evaluation/), [`reports/qwen25vl_tomato_qlora/evaluation/`](reports/qwen25vl_tomato_qlora/evaluation/) |
| 9. Demo app | 16–17 Jul | Streamlit: upload a leaf, get the diagnosis, a confidence and the knowledge-base symptoms. | [`app/`](app/), [`streamlit_app.py`](streamlit_app.py), [`predictions/`](predictions/), [`uploads/`](uploads/) |

A real training example (`artifacts/training/qwen25vl_tomato_qlora/dataset/train.jsonl`):

- **Question:** "What can you observe on this tomato leaf?"
- **Answer:** "This is a tomato leaf. It is affected by bacterial spot, evident from small dark angular water-soaked
  spots. Signs are present, with yellow halos."

## 4. Results to show

From [`reports/qwen25vl_<crop>_qlora/evaluation/metrics.json`](reports/qwen25vl_tomato_qlora/evaluation/metrics.json).

| | Before (zero-shot) | After fine-tuning | Macro-F1 |
|---|---:|---:|---:|
| Tomato, 910 test images | 7.3% | **93.7%** | 0.919 |
| Mango, 200 test images | 17.5% | **82.0%** | 0.811 |

Graphs to show: [`confusion_matrix_finetuned.png`](reports/qwen25vl_tomato_qlora/evaluation/figures/confusion_matrix_finetuned.png)
and [`per_disease_f1.png`](reports/qwen25vl_tomato_qlora/evaluation/figures/per_disease_f1.png) (same names under
`reports/qwen25vl_mango_qlora/` for mango).

Weak spots, worth stating before being asked (`confusion_matrix_finetuned.csv`):

- **Tomato Early Blight:** 33 of 50 correct; 8 called Late Blight, 6 Bacterial Spot.
- **Mango Die Back:** 1 of 25 correct; 6 called Anthracnose, 18 named no disease at all.

## 5. Terms in simple words

| Term | Meaning |
|---|---|
| VLM (vision-language model) | A model that looks at an image and answers in text |
| Zero-shot | Using a model as it is, with no training on our data |
| Fine-tuning | Training it further on our own images and captions |
| LoRA / QLoRA | The big model stays frozen and small extra matrices are trained (40.4 million of 8.29 billion parameters). "Q": the frozen model is stored in 4-bit, so it fits on a 24 GB Mac |
| MLX | Apple's machine-learning framework for M-series chips; CPU and GPU share memory, so no NVIDIA GPU is needed |
| Macro-F1 | F1 averaged over the classes, so small classes count as much as large ones |
| BLEU, ROUGE, CIDEr | Word overlap with the reference captions |
| BERTScore | Similarity of meaning with the reference captions |

## 6. Questions that usually come up

**Why not let a general model write the captions?**
The zero-shot tests showed that general models get the disease wrong, and their mistakes would become training
data. Captions compiled from a cited knowledge base are traceable to a source and rebuild identically every time.

**Why is BLEU only 0.19 when accuracy is 94%?**
BLEU counts exact word sequences shared with the reference caption, and the model words things differently.
BERTScore, which compares meaning, is 0.907.

**Is the test data separate from the training data?**
Yes. The split is fixed (seed 20260711), and the 910 tomato and 200 mango test images were not used for training.

**What are the limitations?**
- PlantVillage photos are lab images on plain backgrounds; field photos would score lower.
- One epoch, with a constant learning rate of 1e-4: the config asks for a cosine schedule, but the training command
  did not apply it (see [`training_plan.md`](reports/qwen25vl_tomato_qlora/training_plan.md) and the log).
- The July adapter files were lost. Both models were retrained in November 2026 on the same data with the same
  config; the downloadable weights are the retrained ones: tomato 90.9% (a snapshot chosen on the validation split;
  weaker than July on spider mites, leaf mold, mosaic virus and target spot), mango 86.0% (see
  [Adapters](README.md#adapters) and [`retrained_2026-11/`](retrained_2026-11/README.md)). It sometimes answers a
  healthy leaf without naming any class (for example "This tomato leaf.", seen with `plantdx infer`), which the demo
  shows as "unknown" (no disease named).

**How does this connect to the ICAIN 2026 paper?**
The same `plantdx` code later became [`projects/10_tih_tomato_qwen_qlora`](../10_tih_tomato_qwen_qlora) (private TIH
tomato data, NVIDIA workstation, 92.76% test accuracy), which is the work in the ICAIN paper.

## 7. Live demo

On a Mac with Apple Silicon, download the retrained adapters first (`python setup/download_models.py 09_plantdx` from
the repository root, 2 × 162 MB), then run the demo or `plantdx infer` (commands in [`README.md`](README.md)).

**Start the demo** (environment `vlm`, created as in [How to start](README.md#how-to-start-mac-with-apple-silicon)):

```bash
conda activate vlm
cd projects/09_plantdx_mlx_knowledge_grounded
python -m streamlit run streamlit_app.py      # opens http://localhost:8501
```

Pick the crop in the sidebar, then drop leaf photos on the Diagnose tab (photos are in
`tomato/raw/PlantVillage/val/<class>/` and `mango/raw/MangoLeafBD/<class>/`; the held-out test images, never used in
training, are the file names listed in `artifacts/training/<run>/dataset/test.jsonl`). The first photo of a crop takes a few
seconds longer (the model loads once); after that about 1–2 s per photo. Remove the photos from the upload box before
switching crops, otherwise they are diagnosed again with the other crop's model.

**What happens after a photo is uploaded** (all in `app/`, which reuses the `plantdx` package in `src/`):

1. `ui.py`, `_process_one`: checks the file type and that the bytes decode as an image (`_validate_image`), shows the
   photo, and reuses the earlier result if the same file was already diagnosed in this session.
2. `inference.py`, `load_for_crop` → `get_model_handle`: loads the base model
   `mlx-community/Qwen2.5-VL-7B-Instruct-4bit` with the crop's adapter from `checkpoints/qwen25vl_<crop>_qlora/`
   (`plantdx.training.inference.load_model` → `mlx_vlm.load`), once per crop, and keeps it in memory
   (`st.cache_resource`).
3. `inference.py`, `run_inference`: builds the chat prompt "Describe the visible condition of this <crop> leaf." with
   the image (`apply_chat_template`), and `mlx_vlm.stream_generate` writes the caption token by token with greedy
   decoding (temperature 0, at most 128 tokens). The Qwen2.5-VL image processor turns the photo into image tokens.
4. Confidence: the average probability the model gave to each token it wrote (`_selected_token_prob`).
5. `classification.py`, `classify`: finds the disease named in the caption with the evaluation's own parser
   (`plantdx.evaluation.classification.extract_disease_id`), with an alias fallback (`dieback`, `sooty mold`, …);
   no disease name → "unclassified".
6. `inference.py`, `_status` and `_disease_info`: a named disease at or above the confidence threshold (sidebar,
   default 0.55) is "confident", below it "low confidence", no name "unknown"; the symptoms shown are read from the
   Disease Knowledge Base (`knowledge_base/dkb.json`), not generated by the model.
7. `storage.py`: the photo is saved under `uploads/<crop>/<class or unknown>/`, with a `.json` record next to it, a
   full record in `predictions/<time>.json`, a line in `logs/predictions.jsonl`, and the sidebar history
   (`history.py`). The result card and the JSON / Markdown download buttons come from `components.py`.

The **Adapter & model** tab shows the adapter's rank, modules, trainable parameters and checksum (compare with
[Adapters](README.md#adapters)); the **Evaluation** tab shows the test-set evaluation of the adapter file that is
loaded (it matches the file by checksum), e.g. tomato 90.9% and mango 86.0% for the downloadable weights.

What can be shown on any computer, without the adapters:

- The code files in section 3.
- The evaluation figures and reports in `reports/`.
- Predictions saved by the demo app on 16 Jul in [`predictions/`](predictions/).
- The raw zero-shot answers of every model in
  [`earlier_experiments/03_zero_shot_4vlm_benchmark_2026-06-25/results/`](earlier_experiments/03_zero_shot_4vlm_benchmark_2026-06-25/results/).
