# Source and Fact-Check Notes

Deliverables covered:

- `Final_Project_Completion_Report.docx`: TIH Project Completion Report, 23 pages, A4
- `Final_Project_Review_Presentation.pptx`: Final review deck, 16 slides, 16:9 (13.333 × 7.5 in)

Revision 2, 1 October 2026: technical content restricted to a single repository branch (see §0).

---

## 0. Technical source of truth (Revision 2)

**All technical numbers, configurations, dataset details, results, figures and implementation details come from one source only:**

> `https://github.com/iAakash1/experimentation`, branch **`refactor/professional-research-layout`**, commit **`66a40bc`** (17 Sep 2026)

The branch was re-fetched from GitHub on 1 Oct 2026 and inspected in full: 417 tracked files, including every document, configuration, result file, figure, script and the relevant source code. The branch has no `tomato/` directory, and `mango/` and `artifacts/` contain only placeholders and READMEs; datasets and checkpoints are gitignored by design (`public_release_manifest.yaml`).

**Not used for any technical value:** `iAakash1/VLM`, the `main` branch, `feat/*` branches, or any other repository or cached copy.

The ICAIN paper is used for the title, publication details, authorship, methodology wording, scientific framing and conclusions. Every ICAIN number used in the deliverables matches the branch's result files exactly.

## 1. Sources

| # | Source | Use |
|---|---|---|
| S1 | `iAakash1/experimentation` @ `refactor/professional-research-layout` (`66a40bc`) | **Sole source of technical values** |
| S2 | Accepted paper PDF: "Efficient QLoRA Fine-Tuning of Qwen2.5-VL for Multi-Class Tomato Disease Classification and Cross-Domain Evaluation" | Title, authors and affiliations, corresponding author, methodology wording, framing, limitations, conclusions |
| S3 | ICAIN 2026 acceptance notification (screenshot, 1 Oct 2026) | Submission ID 953, venue, indexing statement, "top 20%" statement, registration deadline (10 Oct 2026) |
| S4 | Fellowship Crop Research Documentation Reports (Aakash, Kamaaksha; v1.0, 15 Sep 2026) | Non-technical context only: fellowship dates, mentor, project title, TIH mentor names, month of early VLM screening (Jun–Jul 2026), qualitative names and outcomes of the early screening, and the soybean support activity. **No numbers from S4 are used.** |
| S5 | Janhvi/Sheshank completion report and deck; TIH report and PPT templates | Format, structure and visual language only; source of the logos |
| S6 | User brief | Names, emails, phone numbers, mentor, institute, duration, funding figures |

## 2. Branch files used (S1)

| File | Values used |
|---|---|
| `experiments/tih_tomato_qwen/results/internal/metrics.json` | Accuracy 0.9276, macro P 0.9271, macro R 0.9557, macro F1 0.9378, weighted F1 0.9286, latency 0.327 s, 649 images, 11 classes, 0 unparseable |
| `experiments/tih_tomato_qwen/results/internal/per_class_metrics.csv` | Per-class support, precision, recall and F1 (internal) |
| `experiments/tih_tomato_qwen/results/internal/confusion_matrix.csv` | 602 correct (diagonal sum); Late Blight → Target Leaf Spot 13, Leaf Curl 8, Leaf Miner 7 |
| `experiments/tih_tomato_qwen/results/internal/training_summary.json` | 1,908 steps; 2.909 epochs; 11,562.6 s (= 192.7 min); 10.57 GB; 40,370,176 trainable parameters; final loss 8.116e-05; 5,247/649/649 |
| `experiments/tih_tomato_qwen/results/external/metrics.json` | Accuracy 0.69318, macro F1 0.71955, macro P 0.76085, macro R 0.71837, weighted F1 0.69375, latency 0.3728 s, 176 images, 10 classes, folder mapping (Healthy → Healthy Leaf), ignored folders |
| `experiments/tih_tomato_qwen/results/external/per_class_metrics.csv`, `confusion_matrix.csv` | Per-class external values; Late Blight F1 14.29% and recall 10%; 11 of 20 predicted as Target Leaf Spot; 122 correct |
| `experiments/tih_tomato_qwen/experiment.yaml` | Dataset name, hardware (RTX 4500 Ada, 24 GB), configuration summary, canonical config |
| `configs/train/qwen25vl_tih_tomato_moreaccurate.yaml` | Final ICAIN configuration (NF4, double quantization, BF16, 532 px, max sequence 1,024, r 64, α 128, dropout 0.05, q/k/v/o, frozen vision, LR 4e-5, cosine, 3% warm-up, paged AdamW 8-bit, batch 1 × 8, seed 20260911, 80/10/10, prompt text) |
| `configs/train/qwen25vl_tih_tomato.yaml`, `..._strong.yaml` | Configuration progression r16/448 px → r32/392 px (LR 5e-5) → r64/532 px (LR 4e-5) |
| `configs/company_tomato_classes.yaml`, `assets/metadata/label_map.json` | 11 TIH class names |
| `docs/COMPANY_TRAINING.md` | 7,095 raw → 6,545 usable → 5,247/649/649; validation and checkpoint-selection rules; smoke test; 3-hour wall-clock guard; checkpoint contents; test identity bound to the chosen adapter |
| `src/plantdx/company/dataset.py`, `modeling.py`, `train.py`, `external_evaluate.py`, `preflight.py` | Image verification, SHA-256 dedup, contradictory-group removal; 28-px patch rounding, bicubic, −100 answer-only masking; `1/sqrt(n_c)` sampler, patience 1, `3 * 3600` guard; `torch.inference_mode()`; preflight checks (target GPU, BF16, class folders, frozen split integrity, processor, smoke test) |
| `experiments/tih_tomato_qwen/graphs/*.png` | Report Figures 1–4 (unchanged) |
| `research/zero_shot/results/baseline_summary.csv`, `research/mlx_pipeline/results/benchmark_summary.csv`, `RESULTS.md` | MLX tomato (n = 910): 7.3% → 93.7%, macro F1 0.919, weighted F1 0.937. Mango (n = 200): 17.5% → 82.0%, 0.811 / 0.811 |
| `docs/EVALUATION.md` | MLX tomato split 16,201 train / 910 test |
| `docs/TRAINING.md`, `CHANGELOG.md`, `research/mlx_pipeline/configs/*`, `research/mlx_pipeline/experiments/track.yaml` | MLX: Qwen2.5-VL-7B-Instruct-4bit via mlx-vlm on Apple Silicon (M4 Pro, 24 GB); r 16, α 32; LR 1e-4; 1 epoch; batch 1 × 8; 448 px; 90/5/5 image-grouped, disease-stratified split. Tomato 10 classes, mango 8 classes; the mango run uses the identical recipe |
| `research/mlx_pipeline/graphs/*.png` | Report Figures 5–6 (unchanged). Per-disease F1 labels: tomato Early Blight 0.79 and Target Spot 0.85; mango Die Back 0.08 and Anthracnose 0.71. Mango class names |
| `NOTICE`, `CITATION.cff`, `public_release_manifest.yaml` | PlantVillage tomato subset and MangoLeafBD; DKB compiled from cited, authoritative plant-pathology sources; no VLM/LLM or image analysis in the caption path; DKB held in a private archive |
| `tests/unit/{ontology,vocabulary,concepts}/*determinism.py`, `src/plantdx/knowledge_base/models.py` | "18-condition" DKB; "all 46 fields" per DKB entry |
| `docs/ARCHITECTURE.md`, `src/plantdx/dataset/converters.py` | Image-grounded emitter and per-model converters (Qwen2.5-VL, Qwen3-VL, InternVL3, Gemma-3, MLX) remain Milestone-4 stubs |
| `CHANGELOG.md` | Observability and severity-honesty validation rules; Streamlit demo app |

## 3. Values removed or replaced in Revision 2 (absent from the branch)

| Previously used | Earlier source | Action |
|---|---|---|
| Tomato MLX dataset "18,006 images" | `main` README | Replaced with branch values: 16,201 train / 910 test (`docs/EVALUATION.md`) |
| Mango "4,000 images (500 per class)" | `main` README | Replaced with "200 test images, image-grouped 90/5/5 split" |
| Caption-quality scores (BLEU-4, ROUGE-L, METEOR, CIDEr, BERTScore) | `main` README | Removed. The report's caption-quality table and the BERTScore column on slide 13 were deleted |
| MLX per-class F1 0.786 / 0.845 / 0.077 / 0.708 | `main` README | Replaced with the branch figure labels 0.79 / 0.85 / 0.08 / 0.71 |
| "Die Back symptoms overlap heavily" explanation | `main` README | Removed |
| DKB sources "APS, extension services, CABI, FAO" | `main` DKB_report | Replaced with "cited, authoritative plant-pathology sources" (`NOTICE`) |
| 8.29 B base parameters and "≈0.49% trainable" | Paper only | Removed; now "40,370,176 trainable LoRA parameters, pretrained weights frozen" |
| Per-class train/validation split table; "Late Blight 1,561 vs Healthy Fruit 78; ≈20 : 1" | Paper Table 1 only | Replaced with per-class internal/external test support from the branch CSVs (Late Blight 195 vs Healthy Fruit 9) |
| Intermediate "strong" run, 87.98% accuracy / 0.8991 macro F1 | Fellowship reports | Replaced with "Not reported in the repository" (the configuration itself remains, from the branch) |
| Early exploration numbers: 98.50%, 0.9827, BLEU-1 0.2081, ROUGE-L 0.3017, "7 VLMs / 30 images / 6 classes / 16 of 30", CLIP threshold 0.22, LoRA r32/α64, "PlantDx v1", RTX 4500 Ada for that stage | `iAakash1/VLM` and fellowship reports | Removed. Only qualitative model-screening outcomes (from S4) remain, labelled as not part of the repository |
| Reference to `iAakash1/VLM`; multi-branch reference | — | Removed; the reference now cites the single branch and commit |

## 4. Important numbers (all from S1)

| Quantity | Value |
|---|---|
| TIH raw → retained images | 7,095 → 6,545 (92.25%, derived); 550 excluded (derived) |
| Frozen split (seed 20260911) | 5,247 / 649 / 649 (80/10/10) |
| Internal (649 images, 11 classes) | Accuracy 92.76%, macro F1 93.78%, macro recall 95.57%, macro precision 92.71%, weighted F1 92.86%, 0.327 s/image, 602 correct |
| External (176 images, 10 classes) | Accuracy 69.32%, macro F1 71.95%, macro precision 76.08%, macro recall 71.84%, weighted F1 69.38%, 0.373 s/image, 122 correct |
| Gap (derived) | −23.44 pp accuracy; −21.83 pp macro F1 |
| Training | 1,908 steps; 2.909 epochs; 192.7 min; 10.57 GB; 40,370,176 trainable parameters; final loss 8.12 × 10⁻⁵ |
| MLX tomato (PlantVillage subset, 10 classes) | 16,201 train / 910 test; 7.3% → 93.7%; macro F1 0.919; weighted F1 0.937 |
| MLX mango (MangoLeafBD, 8 classes) | 200 test; 17.5% → 82.0%; macro F1 0.811; weighted F1 0.811 |
| Funding (S6) | ₹2,40,000 sanctioned (₹1,20,000 per fellow); ₹10,000/month/fellow; ₹20,000 contingency (overall); ₹96,000 received per fellow (₹1,92,000 total); ₹48,000 balance |
| Publication (S2, S3) | ICAIN 2026; ID 953; Proceedings of ICAIN 2026 – Springer LNNS Series; accepted; "top 20%" statement; registration before 10 Oct 2026 |

**Automated audit (Revision 2).** Every numeric token in both files was extracted and checked against the branch text. The only tokens not literally present in the branch are:

- values derived arithmetically from branch values (550, 92.25, gaps, 192.7, 602, 122);
- labels on the branch's own PNG figures (0.79, 0.85, 0.08, 0.71);
- a rounding of a branch value (8.12 from 8.116e-05);
- non-technical numbers (section and page numbers, dates, funding, phone numbers, ID 953, the model name "OPT-2.7B").

## 5. How the experiments are kept separate

1. **Track A: TIH tomato (ICAIN).** `experiments/tih_tomato_qwen`.
2. **Track B: PlantDx MLX research.** `research/mlx_pipeline` and `research/zero_shot`.
3. **Track C: early exploration.** Qualitative only; no numbers are quoted.

The report's Table 1 and slide 8 state that the tracks are not directly comparable.

## 6. Unresolved information (fill in or confirm)

1. **Project grant number.** Not in any source; the cover sheet shows a bracketed placeholder.
2. **ICAIN 2026 conference date, DOI, volume and pages.** Not available; reported as "not stated" or "not yet assigned".
3. **Approved proposal text.** The objectives and deliverables tables were reconstructed from the title and the documented work. Check them against the sanctioned proposal.
4. **Previous-review action items.** None were in the sources, so slide 2 shows a "Team & Project Area" card in that slot.
5. **Head-wise budget split.** It cannot be reconstructed from the given totals, and instalment dates are unknown. The contingency is shown as *planned* only.
6. **Fellows' formal qualification.** Only the department (Computer Engineering, PCCOE) is known.
7. **Awareness or training camps.** Stated as "none organised" for lack of evidence. Edit if applicable.
8. **Dec 2025 – May 2026 activities.** Not itemised in the sources, so they are described at a high level only.
9. **Mango dataset total size and per-class TIH train/validation counts.** Not recorded in the branch, so they are not stated.
10. **Milestone months.**
    - Jun–Jul 2026 screening comes from the fellowship reports.
    - Jul 2026 (PlantDx) and Sep 2026 (TIH) come from commit dates in the branch history.
    - Aug–Sep 2026 comes from the dataset name `Tomato_12thAug26` and the Sep 2026 commits.
    - October 2026 comes from the ICAIN notification.

## 7. Assumptions deliberately avoided

- No DOI, volume, pages, conference date, patent, award, "best paper" claim or commercialisation claim.
- No claim that the contingency was spent. No head-wise expenditure entries. No instalment dates.
- The ₹48,000 balance is described as the remaining sanctioned amount, not as an unpaid stipend.
- No field-deployment or farmer-impact claims; the Streamlit app is described as a research prototype.
- No values from `iAakash1/VLM`, other branches, generic Qwen2.5-VL benchmarks or the reference project.

## 8. Deviations from the reference format (intentional, unchanged from Revision 1)

- Cover-sheet rows are numbered 1–7; the template's numbering is erroneous.
- Part A 2(b) adds an "experiment tracks" table.
- Part B adds short Conclusion, Acknowledgements, References and Annexures sections.
- The signature block has three signatories.
- Slide 2 shows "Team & Project Area" instead of "Action Items".
- There is no Outline slide, as in the reference deck.

## 9. Editability and file compatibility

### Revision 3: file-opening fix

- **Problem.** The previous PPTX could not be opened in some tools. The cause was the speaker notes: python-pptx's generated notes master and notes slides make Apple's Office importer (used by Keynote, Pages and Preview) reject the file. A plain deck opened; the same deck plus a single speaker note did not. The chart axis IDs were also negative, which violates the OOXML schema (`xs:unsignedInt`).
- **Fixes:**
  - Speaker notes removed. Per-slide source citations remain in this file.
  - Chart axis IDs made positive.
  - Slide size set to exactly 12192000 × 6858000 EMU (16:9), with the stale 4:3 `type` attribute removed.
  - The title slide's decorative arc replaced with a ring, which every renderer draws.
- **DOCX hardening:**
  - Cover colour bar is now a shaded paragraph rather than an empty table.
  - Page starts use "page break before" instead of separate break paragraphs, so Word shows no blank line at the top of a page.
  - Word compatibility mode raised to 15 (modern Word, no "Compatibility Mode" banner).
  - `updateFields` is set, so Word refreshes the TOC and page fields on open. Click **Yes** when Word asks.
- **Verification:**
  - Every XML part of both files passes strict ISO/IEC 29500 schema validation (PPTX 37 parts, DOCX 9 parts; 0 errors).
  - Both files pass the skill validators.
  - Both open in Apple's Office importer; each of the 16 slides was rendered individually through it.
  - Both open and render in LibreOffice.

**DOCX.**
- Built-in heading styles and native tables.
- An updatable TOC field pre-filled with verified page numbers.
- SEQ caption fields; PAGE and NUMPAGES fields in the header and footer; the template's page border.
- Figures 7–10 are editable table diagrams. Figures 1–6 are the branch's PNG figures, unchanged.

**PPTX.**
- 16 slides at 13.333 × 7.5 in.
- Native text boxes and shapes, 3 native tables and 4 native charts.
- The only pictures are the TIH and IIT Bombay logos. No groups and no screenshots.
- Arial throughout. No speaker notes (see Revision 3).

## 10. Final quality audit (Revision 2)

| Check | Result |
|---|---|
| Names, emails, phones, mentor, institute, title, dates, funding | Unchanged and verified |
| Technical numbers come only from `iAakash1/experimentation` @ `refactor/professional-research-layout` | Yes (automated token audit, §4) |
| No `iAakash1/VLM`, `main`-branch or other-source technical values | Yes |
| ICAIN numbers equal the branch results | Yes |
| Internal and external distinguished; cross-domain limitation shown; Tracks A/B/C distinguished | Yes |
| Layout and structure unchanged apart from the edits listed in §3 | Yes. The report is 23 pages, the TOC was re-verified, and the caption-quality table was removed |
| Both files validate and were re-rendered for visual QA | Yes |
