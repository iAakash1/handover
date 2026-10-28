<!-- Text and figures extracted from `Final_Project_Review_Presentation.pptx` so it can be read on GitHub; the original file is next to this one. -->

## Slide 1

![](Final_Project_Review_Presentation_figures/figure_01.png)

![](Final_Project_Review_Presentation_figures/figure_02.png)

- Vision-Language Model Based Plant Disease Diagnosis for Tomato and Mango

- Project Completion Review  |  TIH IIT Bombay CHANAKYA Fellowship

- By
- Kamaaksha Rajendra Kharul
- Aakash Trimbak Jawle
- Mentor: Dr. Santwana Gudadhe
- Pimpri Chinchwad College of Engineering, Pune




- Qwen2.5-VL-7B
- + QLoRA
- Tomato  ·  Mango
- Disease & Pest Classification
- ICAIN 2026 · Accepted








## Slide 2


- Project Card


- 2


- Start Date and End Date of the Project
- Start Date:   01 December 2025
- End Date:     30 September 2026


- Awards/Recognition
- Research paper accepted for publication in the Proceedings of ICAIN 2026 – Springer LNNS Series (Submission ID 953).
- Acceptance notification: paper among the top 20% of papers received/reviewed.


- Total Sanctioned
- Amount

- ₹ 2.40

- Lakhs

- ₹1.20 L per fellow


- Total Amount
- Disbursed from TIH

- ₹ 1.92

- Lakhs

- ₹0.96 L per fellow

- Balance against sanction: ₹0.48 Lakh (not yet received)


- Team & Project Area

- Fellows: Kamaaksha Rajendra Kharul, Aakash Trimbak Jawle
- Mentor: Dr. Santwana Gudadhe
- Institute: PCCOE, Pune
- Area: AgriTech — AI/ML, Computer Vision, Vision-Language Models


- IP / Patents Details

- No patent/IP filing was made during the project period.
- Code released under Apache-2.0 (github.com/iAakash1/experimentation); private data and checkpoints excluded.


- Commercialization Plan

- No commercialization was undertaken during the project period.
- Outputs are research workflows and trained adapters.
- Field validation on independently collected images is a prerequisite for any deployment.

## Slide 3



- 1. Background and Problem Statement

![](Final_Project_Review_Presentation_figures/figure_03.png)


- 3


- Domain

- Agriculture Technology (AgriTech)


- Focus Area

- Vision-Language Models for image-based tomato and mango disease & pest classification


- Need for the Project

- Reliable diagnosis needs models that can be adapted efficiently on limited hardware and that are evaluated honestly beyond their own training data.


- Background

- Visually similar disease and pest symptoms complicate manual diagnosis and can delay intervention.
- The TIH tomato data mixes leaf and fruit diseases, pests and healthy classes; mango leaf symptoms (e.g., Die Back) overlap strongly.
- Expert visual inspection is difficult to scale and depends on expert availability.
- General-purpose VLMs fail zero-shot: 7.3% (tomato) and 17.5% (mango) accuracy for base Qwen2.5-VL-7B.
- Full fine-tuning of 7B-parameter VLMs is costly, and accuracy depends strongly on acquisition conditions — in-domain scores may not transfer.


- Problem Statement

- To adapt a general-purpose vision-language model for accurate multi-class tomato and mango disease & pest classification on workstation-class hardware, using audited, reproducible data pipelines, and to quantify how well it generalises to independently collected images.

## Slide 4



- 2. Objectives

![](Final_Project_Review_Presentation_figures/figure_04.png)


- 4







- To develop and rigorously evaluate a parameter-efficient, reproducible vision-language-model workflow for tomato and mango disease and pest classification, including an honest assessment of cross-domain generalisation.



- 01

- Investigate VLMs

- Screen multiple VLM families and select a backbone for crop diagnosis.



- 02

- Curate & Audit Datasets

- Verify images, remove duplicates and label conflicts, freeze splits.



- 03

- Parameter-Efficient Fine-Tuning

- Adapt Qwen2.5-VL-7B-Instruct with QLoRA on a single 24 GB GPU.



- 04

- Tomato Classification

- Train and evaluate an 11-class tomato disease and pest classifier.



- 05

- Mango Classification

- Fine-tune and evaluate an 8-class mango leaf disease model.



- 06

- Cross-Domain Evaluation

- Measure generalisation on an independent external benchmark.



- 07

- Knowledge-Grounded Diagnosis

- Build cited disease knowledge, ontology and controlled vocabulary.



- 08

- Reproducibility & Publication

- Release tested, config-driven code and publish the findings.

## Slide 5



- 3. Expected Benefits of the Project

![](Final_Project_Review_Presentation_figures/figure_05.png)


- 5



- 01

- Automated Disease & Pest Identification

- Image-based classification that reduces dependence on manual inspection.



- 02

- Efficient Use of Large VLMs

- 40.37 M trainable LoRA parameters and 10.57 GB peak memory on one 24 GB GPU.



- 03

- Cleaner, Leak-Free Data

- Audit removed 550 duplicate/contradictory files; no image appears in two splits.



- 04

- Reusable, Reproducible Workflows

- Config-driven pipelines, fixed seeds, split hashes, preflight checks and tests.



- 05

- Honest Robustness Assessment

- Internal vs external evaluation pinpoints weak classes (e.g., Late Blight).



- 06

- Extensible to More Crops

- The same recipe was applied to tomato and mango and can be reused for new crops.


- Overall Benefit


- A reproducible, parameter-efficient VLM workflow for tomato and mango disease classification, with transparent evaluation that shows what data is needed before future field deployment.

## Slide 6



- 4. Major Milestones

![](Final_Project_Review_Presentation_figures/figure_06.png)


- 6


- 01


- Project Initiation


- Dec 2025

- Fellowship began 1 Dec 2025
- Problem scoping; dataset & model exploration


- 02


- VLM Screening


- Jun 2026

- LLaVA-1.5/1.6, BLIP-2, Florence-2 trialled
- Qwen2.5-VL-7B selected as backbone


- 03


- Zero-shot & OOD


- Jun–Jul 2026

- Multi-VLM zero-shot benchmark
- CLIP-based OOD screening explored


- 04


- Knowledge-Grounded PlantDx


- Jul 2026

- Cited knowledge base (18 conditions)
- Ontology, vocabulary, caption corpus


- 05


- MLX QLoRA: Tomato & Mango


- Jul 2026

- Tomato 7.3% → 93.7%
- Mango 17.5% → 82.0%
- Streamlit demo


- 06


- TIH Dataset Audit & QLoRA


- Aug–Sep 2026

- 7,095 → 6,545 images
- CUDA QLoRA; LoRA r 32 → 64


- 07


- Internal + External Evaluation


- Sep 2026

- Internal: 92.76% accuracy
- External: 69.32% accuracy


- 08


- ICAIN 2026 Paper


- Oct 2026

- Accepted for publication
- Springer LNNS · ID 953


- Project
- Progression


- VLM
- Exploration


- Knowledge-
- Grounded PlantDx


- MLX
- Fine-Tuning


- TIH QLoRA
- Benchmark


- Cross-Domain
- Evaluation


- Publication




- Project Closure
- 30 September 2026

## Slide 7



- 5. Shortfalls and Reasons in Achievements of Milestones

![](Final_Project_Review_Presentation_figures/figure_07.png)


- 7



- i

- Core milestones were completed; the items below required additional iterations or remain open, and are reported transparently.



- 01

- Model Selection
- (Early VLMs)

- LLaVA-1.5 / 1.6: batch-collation incompatibility
- Florence-2: environment issues; BLIP-2: unreliable structured output


- Impact
- Extra iterations before converging on Qwen2.5-VL-7B-Instruct.



- 02

- Cross-Domain
- Generalisation

- External accuracy 69.32% vs 92.76% internal
- Late Blight external F1 14.29% (11 of 20 → Target Leaf Spot)


- Impact
- Not yet field-ready; diverse, field-acquired data is needed.



- 03

- Compute & Time
- Budget

- 3-hour wall-clock safeguard ended training at 2.909 of 3 epochs
- Single seed; no large hyper-parameter search


- Impact
- Results come from one run; repeated seeds are planned.



- 04

- Pending Scope
- Items

- Image-grounded instruction builder and multi-model converters are still stubs
- Mango Die Back F1 0.08 (MLX track)


- Impact
- Carried forward into future work.


- Overall: addressed through iterative configuration changes (LoRA r 32 → 64, image side 392 → 532 px) and transparent reporting of limitations.

## Slide 8



- 6. Technical Achievements — Overall Approach & Datasets

![](Final_Project_Review_Presentation_figures/figure_08.png)


- 8


- Dataset


- Data Audit


- Cleaning /
- Dedup


- Dataset
- Construction


- VLM
- (Qwen2.5-VL)


- QLoRA
- Fine-Tuning


- Validation


- Internal
- Test


- External
- Benchmark


- Cross-Domain
- Analysis

- Data preparation

- Model adaptation

- Evaluation



- Track A · TIH Tomato (ICAIN 2026)

- Dataset

- Tomato_12thAug26 (private, TIH)

- Classes

- 11 — diseases, pests, healthy; leaf & fruit

- Images

- 7,095 raw → 6,545 retained

- Split

- 5,247 / 649 / 649 (80/10/10)

- External

- 176 images, 10 classes

- Hardware

- NVIDIA RTX 4500 Ada, 24 GB



- Track B · PlantDx Tomato (MLX)

- Dataset

- PlantVillage tomato subset (public)

- Classes

- 10 leaf conditions

- Train split

- 16,201 images

- Test split

- 910 (image-grouped)

- Supervision

- Knowledge-grounded captions

- Hardware

- Apple Silicon, 24 GB unified



- Track B · PlantDx Mango (MLX)

- Dataset

- MangoLeafBD (public)

- Classes

- 8 leaf conditions

- Split

- 90/5/5, image-grouped

- Test split

- 200 images

- Supervision

- Knowledge-grounded captions

- Hardware

- Apple Silicon, 24 GB unified


- Important: tracks use different datasets, hardware and protocols — results are reported separately and are not directly comparable.

## Slide 9



- 6. Dataset Integrity & Reproducibility (TIH Tomato)

![](Final_Project_Review_Presentation_figures/figure_09.png)


- 9


- 7,095 raw labelled images

- 11 class folders · read-only



- Image verification

- supported extension + decodable image



- SHA-256 exact deduplication

- same-class copies collapsed to one



- Contradictory duplicates removed

- hash under >1 label → group excluded



- 6,545 retained (92.25%)

- 550 files excluded



- 5,247

- Train


- 649

- Validation


- 649

- Test

- Class-stratified 80/10/10 · seed 20260911 · split hashes frozen

- Strong class imbalance (Late Blight 195 vs Healthy Fruit 9 test images) → inverse-square-root sampling


- Reproducibility Controls


- 1

- Deterministic split
- seed + image hash ordering, not file-system order


- 2

- Frozen split identities
- hashes checked; changes block training


- 3

- Deep preflight
- target GPU / BF16, class folders, frozen-split integrity, processor


- 4

- QLoRA smoke test
- save → resume → reload adapter verified


- 5

- Checkpoint metadata
- adapter, optimizer, scheduler, RNG and trainer state


- 6

- Test identity
- bound to the chosen adapter; test evaluated once

## Slide 10



- 6. Model & QLoRA Methodology

![](Final_Project_Review_Presentation_figures/figure_10.png)


- 10


- Tomato image

- RGB · ≤532 px side
- 28-px aligned



- Vision encoder

- FROZEN



- Qwen2.5 language model

- 4-bit NF4 (frozen) + LoRA on q/k/v/o



- Class label

- e.g. “Late Blight”


- Fixed prompt: “Classify this tomato plant image into exactly one of these categories: Aphid, … , White Fly. Return only the category name.”

| Component | Setting |
|---|---|
| Base model | Qwen2.5-VL-7B-Instruct |
| Quantisation | 4-bit NF4, double quant., BF16 compute |
| LoRA | r 64 · α 128 · dropout 0.05 · q/k/v/o only |
| Trainable parameters | 40,370,176 (pretrained weights frozen) |
| Optimiser | Paged AdamW 8-bit · LR 4e-5 · cosine, 3% warm-up |
| Batch | 1 × 8 gradient accumulation (effective 8) |
| Epochs / seed | 3 planned (2.909 run) / 20260911 |



- Class-Balanced Sampling

- w_c = 1 / √n_c

- Reduces Late Blight dominance without over-repeating tiny classes such as Healthy Fruit.



- Answer-Only Supervision

- prompt tokens → −100

- Loss is computed only on the class-label tokens → 0 unparseable outputs on both benchmarks.



- Validation-Based Selection

- macro F1 → accuracy → loss

- Early stopping on validation loss (patience 1); final adapter chosen by validation macro F1.

## Slide 11



- 6. Experimental Results — Internal Test (TIH, 649 images)

![](Final_Project_Review_Presentation_figures/figure_11.png)


- 11


- 92.76%

- Accuracy

- 602 of 649 correct


- 93.78%

- Macro F1

- primary metric (imbalance)


- 95.57%

- Macro Recall

- macro precision 92.71%


- 0.327 s

- Latency / image

- single RTX 4500 Ada GPU


- 1,908

- optimizer steps

- 2.909

- epochs


- 192.7 min

- training time


- 10.57 GB

- peak reserved memory


- 40.37 M

- trainable LoRA parameters


- 0

- unparseable outputs



- Class-Level Observations

- Late Blight (195 test images): F1 90.91%, recall 84.62% — confused with Target Leaf Spot (13), Leaf Curl (8), Leaf Miner (7).
- Target Leaf Spot: 100% recall but 68.75% precision (lowest F1, 81.48%).
- Healthy Fruit: 100% on only 9 test images — small support.

## Slide 12



- 6. Cross-Domain Evaluation (Internal vs External)

![](Final_Project_Review_Presentation_figures/figure_12.png)


- 12

| Metric | Internal test | External |
|---|---|---|
| Images | 649 | 176 |
| Classes | 11 | 10 |
| Accuracy | 92.76% | 69.32% |
| Macro F1 | 93.78% | 71.95% |
| Weighted F1 | 92.86% | 69.38% |
| Latency | 0.327 s | 0.373 s |


- −23.44 pp

- accuracy


- −21.83 pp

- macro F1


- “Cross-domain evaluation revealed a substantial generalization gap, highlighting the importance of broader and more diverse training data.”
- Late Blight F1 fell from 90.91% to 14.29% (11 of 20 external images predicted as Target Leaf Spot), while Mealy Bugs (93.33%) and White Fly (91.67%) transferred well — the shift is class-dependent.

## Slide 13



- 6. PlantDx: Knowledge-Grounded Research Track

![](Final_Project_Review_Presentation_figures/figure_13.png)


- 13


- Disease Knowledge Base

- 18 cited conditions (10 tomato, 8 mango)



- Ontology Compiler

- typed, evidence-linked knowledge graph



- Vocabulary + Symptom Lexicon

- controlled, traceable terminology



- Concepts + Templates → Caption Corpus

- deterministic, multi-stage validated



- Audited & Normalised Images

- PlantVillage tomato · MangoLeafBD



- QLoRA Fine-Tuning (MLX)

- Qwen2.5-VL-7B-4bit · r 16 · Apple Silicon



- Evaluation + Streamlit Demo

- base vs fine-tuned on frozen test splits

| Crop | Classes | Test images | Macro F1 | Weighted F1 |
|---|---|---|---|---|
| Tomato | 10 | 910 | 0.919 | 0.937 |
| Mango | 8 | 200 | 0.811 | 0.811 |


- Principles: cited knowledge only; no VLM/LLM or image analysis in the caption path; observability- and severity-honest; bit-for-bit reproducible.
- In-distribution scores on public single-leaf datasets — separate from the TIH benchmark.

## Slide 14



- 7. List of Publications, Patents etc.

![](Final_Project_Review_Presentation_figures/figure_14.png)


- 14



- ICAIN 2026


- Accepted for publication

- Efficient QLoRA Fine-Tuning of Qwen2.5-VL for Multi-Class Tomato Disease Classification and Cross-Domain Evaluation

- Aakash Trimbak Jawle, Kamaaksha Rajendra Kharul, Santwana Gudadhe, Harish Lokawad, Rutuja Chavan

- Proceedings of ICAIN 2026 – Springer LNNS Series


- Submission ID

- 953


- Status

- Accepted


- Registration

- before 10 Oct 2026

- The acceptance notification states that the paper is among the top 20% of the papers received/reviewed.
- Indexing as stated in the notification: zbMATH, Scopus and DBLP. DOI / volume / pages: not yet assigned.



- Patents / IP

- No patent/IP filing was made during the project period.



- Code & Artefacts

- github.com/iAakash1/experimentation, branch refactor/professional-research-layout (Apache-2.0): configs, aggregate metrics, figures, tests. Private data and checkpoints excluded.



- Fellowship Documentation

- Crop Research Documentation Reports (v1.0, 15 Sep 2026) prepared by both fellows.

## Slide 15



- 8. Future Work Planned

![](Final_Project_Review_Presentation_figures/figure_15.png)


- 15



- 01

- Cross-Domain Robustness

- Train on independently acquired, multi-source field images
- Keep the external benchmark untouched; add domain-robustness tests
- Target the Late Blight ↔ Target Leaf Spot confusion first


- 69.32%

- current external accuracy (baseline to improve)



- 02

- Dataset & Crop Expansion

- More samples for rare/confusable classes (Healthy Fruit, mango Die Back)
- Near-duplicate detection and repeated-seed runs
- Extend the recipe to additional crops


- 2 crops

- covered today (tomato, mango)



- 03

- Deployment Optimisation

- Reduce latency and memory with quantised inference
- Evaluate on edge / mobile-class hardware
- Field validation with expert verification before farmer-facing use


- 0.327 s

- current GPU latency / image



- 04

- Reliability & Knowledge Grounding

- Calibrated confidence, uncertainty handling and OOD rejection
- Complete the image-grounded instruction builder
- Compare Qwen3-VL, InternVL3 and Gemma-3


- 3

- further VLM families to benchmark

## Slide 16



- 9. Conclusion

![](Final_Project_Review_Presentation_figures/figure_16.png)


- 16







- The project built and evaluated parameter-efficient VLM workflows for tomato and mango disease classification — from audited data to QLoRA-adapted Qwen2.5-VL-7B and transparent internal and cross-domain evaluation — and the work has been accepted at ICAIN 2026.

- Key Takeaways


- 01

- What Was Built

- Audited datasets, CUDA and MLX QLoRA pipelines, evaluation framework, knowledge-grounded PlantDx.


- 02

- Measurable Results

- TIH internal: 92.76% accuracy, 93.78% macro F1. PlantDx: tomato 93.7%, mango 82.0%.



- 03

- Publication

- Accepted in the Proceedings of ICAIN 2026 – Springer LNNS Series (ID 953).



- 04

- Key Limitation

- External benchmark: 69.32% accuracy; Late Blight F1 14.29% — a clear domain gap.



- 05

- Next Direction

- Broader, field-representative data and robustness-focused evaluation.



- Thank You

- Kamaaksha Rajendra Kharul  ·  Aakash Trimbak Jawle
- Mentor: Dr. Santwana Gudadhe  ·  Pimpri Chinchwad College of Engineering, Pune
