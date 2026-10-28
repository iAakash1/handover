# 05 — Quality Assurance Protocol (Task 8)

**Deliverable 6 of 8.** Defines how the generated caption library is reviewed before it is accepted into Stage 3: the sampling plan (how many, how stratified), the reviewer checklist, the hallucination taxonomy and its detection, inter-annotator agreement, and batch acceptance rules.

Central principle — **the division of labor between validators and humans:**
- **Generation is image-blind** (invariant #1): the generator never sees pixels, which is what guarantees no label leakage / no VLM circularity.
- **Automated validators (doc 03) are image-blind too**: they enforce consistency with the *label's licensed description*.
- **Human QA is image-aware**: reviewers look at the actual image. They exist to catch the *single* error class the image-blind pipeline cannot — a caption that is DKB-legal for the disease but **over-claims a characteristic feature that is not actually visible in this specific frame** (the "disease-typical but not-in-this-image" risk from doc 00 §5). Everything else the validators should already guarantee; QA also *audits whether they did*.

---

## 1. Two-tier review process

### Tier 1 — Calibration / pilot (before any bulk generation is accepted)
- Generate a **bootstrap batch**: `30 captions × 18 diseases = 540`, using the full style/task distribution.
- **100% human review** by ≥2 reviewers.
- Purpose: (a) tune the DKB-derived ontology, templates, lexicons, and hedging *before* scaling; (b) establish the reviewer checklist calibration and measure inter-annotator agreement (§5); (c) confirm the validator battery's real-world false-negative rate.
- Exit gate: after fixes, a **re-generated** bootstrap batch must show **zero critical defects** and ≥0.8 reviewer kappa. Only then is bulk generation authorized.

### Tier 2 — Acceptance audit (on the full library, per generation run)
- **Stratified random sample** of the full library (`§2`), image-aware review.
- Batch acceptance rule (`§6`) decides release / block.
- Oversample the highest-risk strata (educational/differential captions; the diagnostic confusable-pair split; dense/long captions; secondary-sign hedged captions).

## 2. Sampling plan (how many per disease)

### 2.1 Basis
With `n` reviewed and `d` defects observed, the 95% upper confidence bound on the true defect rate is ≈ `(d + 2)/n` (Agresti–Coull, simplified); for `d=0`, the "rule of three" gives ≈ `3/n`. This sets sample sizes for the confidence we want on *critical* defects being rare.

| n per stratum | 95% upper bound if 0 defects | Use |
|---------------|------------------------------|-----|
| 30 | ~10% | pilot per disease (Tier 1) |
| 50 | ~6% | minimum acceptance audit per disease |
| 100 | ~3% | recommended acceptance audit per disease |
| 200 | ~1.5% | high-stakes / final pre-publication audit |

### 2.2 Plan
- **Per disease:** review **≥100 captions** (18 × 100 = **1,800**) for the acceptance audit, so that "0 critical defects" backs a ≤3% upper bound per disease.
- **Stratification within a disease** (proportional to the generation distribution, doc 02 §6): across `style` (all 8), `task_type` (all valid), `register`, and `hedged` vs not. Guarantee ≥5 from each `dense`, `long`, `educational`, and `hedged` cell (these are higher risk and lower frequency).
- **Risk oversample:** additionally review **100%** of captions on the **diagnostic confusable-pair split** (doc 04 §5) and **100%** of `differential`-task captions for the hardest DKB pairs (early-blight/target-spot/Septoria; anthracnose/bacterial-canker; powdery-mildew/sooty-mould), because these carry the most over-claim and cross-disease risk.
- **Continuous audit:** for any regeneration after a fix, a fresh independent sample (no reuse of previously reviewed items) — never "review until it passes."

## 3. Reviewer checklist (per caption; reviewer sees image + caption + DKB card + provenance)

The review UI (`§7`) shows: the image; the caption; the instruction; the disease's DKB entry (report card); and the caption's provenance (concepts, template, expansion edges). Reviewer answers each item **Yes/No/NA**; any "No" on a critical item fails the caption.

**A. Scientific correctness (vs DKB)** — *critical*
1. Every symptom mentioned is listed for this disease in the DKB (no foreign/invented symptom)?
2. No non-leaf-observable claim (fruit, twig, flower, tree, yield, vascular, gummosis, tear-stain, adult insect)?
3. No forbidden term/adjective for this disease?
4. Correct pest/pathogen framing (no "infection/pathogen/lesion" for pest/surface classes; no "feeding/cut/gall" for true pathogens)?
5. No severity **stage** claim (mild/moderate/severe/early/advanced) unless a severity label was supplied?
6. No rival disease's hallmark term except inside a correct differential clause?

**B. Image grounding (image-aware; the human-only check)** — *major*
7. The asserted **primary sign** is actually visible in this image (not merely disease-typical)? If the caption asserts a specific feature that is plainly absent/contradicted in this frame, mark **major**.
8. Asserted color/shape/location/texture are consistent with what the image shows (not contradicted)?
9. Extent words ("scattered/numerous") are not grossly wrong for this image?

**C. Completeness & consistency** — *critical (10) / major (11)*
10. States the disease identity (or health status) and ≥1 primary sign / healthy_state?
11. Internally consistent (no contradictions; hedging present where a secondary sign is mentioned)?

**D. Language quality** — *minor*
12. Grammatical, natural, correctly punctuated, appropriate length for its style?
13. Not a near-duplicate of another caption of the same image?

## 4. Hallucination taxonomy & detection

| Code | Hallucination type | Which validator should prevent it | What human QA adds |
|------|--------------------|-----------------------------------|--------------------|
| H1 | Non-observable / foreign symptom asserted (fruit lesion, twig canker, pycnidia on wrong disease) | V2 (forbidden-symptom), V3, V8 | Confirms 0 leakage; any H1 in sample ⇒ validator/lexicon bug ⇒ **block release**, fix lexicon, regenerate, re-audit |
| H2 | Cross-disease hallmark term (Septoria "pycnidia" on early blight) | V8, V4 | Same as H1 — must be 0 |
| H3 | **Visual over-claim**: DKB-legal feature not actually present in this image | *none* (validators are image-blind) | **Primary human-only catch** (checklist B). Drives hedging-rate / concept-selection tuning |
| H4 | Invented terminology not in DKB vocab | V4 (closed vocab) | Confirms whitelist is complete |
| H5 | Internal contradiction / mutex violation / severity-stage claim | V9, V10 | Confirms 0; any occurrence ⇒ validator bug |
| H6 | Register error (pest called "infected") | V7 | Confirms 0 |

Interpretation: **H1, H2, H4, H5, H6 must be 0** in any accepted sample — they are things the validators are supposed to make impossible, so a single occurrence is treated as a *validator defect* (halt, fix, regenerate), not a tolerable caption-level error. **H3 is the residual, human-managed risk**; it is bounded by the acceptance rule (§6) and reduced by raising `hedging_probability`, biasing concept selection toward high-prevalence primary signs, and shortening captions for classes whose per-image appearance varies most.

### 4.1 Automated hallucination trace (assist for reviewers)
For each reviewed caption the UI pre-computes: the concept→text alignment (which span realizes which concept) and re-runs V2/V3/V4/V8 live, highlighting any span that trips a check. Reviewers thus spend their attention on **H3** (image grounding), where machines can't help.

## 5. Inter-annotator agreement
- ≥2 reviewers independently review a **20% overlap** of every audit sample.
- Compute **Cohen's κ** on the binary accept/reject decision and on each critical checklist item; target **κ ≥ 0.80**. If κ < 0.80, the checklist wording is ambiguous → refine definitions and re-calibrate (Tier 1 style) before trusting single-reviewer items.
- All disagreements are **adjudicated** by a third reviewer; adjudicated labels are the ground truth for the acceptance computation.
- Record reviewer_id, timestamps, and per-item verdicts in `qa/review_results/`.

## 6. Batch acceptance rule
Per audit sample (per disease and overall):

| Defect class | Definition (checklist items) | Acceptance threshold |
|--------------|------------------------------|----------------------|
| **Critical (C)** | A1–A6, C10 (H1/H2/H4/H5/H6) | **0** in the sample. Any C ⇒ **block release**; fix root cause (usually lexicon/ontology/validator), regenerate affected disease(s), re-audit a fresh sample. |
| **Major (M)** | B7–B9, C11 (H3 visual over-claim, consistency) | **≤ 1%** of the sample per disease. Exceed ⇒ tune hedging/selection, regenerate, re-audit. |
| **Minor (m)** | D12–D13 (language, near-dup) | **≤ 5%** per disease. Exceed ⇒ template/diversity tuning; does not block if C=0 and M within bound, but must be logged. |

Also gate on the **diversity metrics** (doc 00 §7.7): a library that meets QA defect thresholds but fails a hard diversity gate is **not** accepted (it would over-fit sentence structure).

**Release decision:** a library version is accepted iff, on the acceptance audit, every disease has C=0, M≤1%, m≤5%, all diversity hard-gates pass, and the validator run had 0 hard-errors and ≤2% fallback (doc 03 §5). Acceptance is recorded in `qa/acceptance_<library_version>.md` with the sample, counts, κ, and sign-off.

## 7. Tooling & workflow requirements (for the implementer)
- A lightweight local review UI (no external service; runs on the Mac) that, per item, renders: image, instruction, caption, DKB card (from `DKB_report.md`), provenance, live V2/V3/V4/V8 highlights, and a checklist form writing JSONL to `qa/review_results/`.
- Deterministic **sample manifests** (`qa/review_samples/<library_version>.jsonl`) drawn with a recorded seed so audits are reproducible and re-drawable.
- Reviewers work from the manifest; the UI never reveals other reviewers' answers (independence for κ).
- All QA outputs are versioned against `library_version`, `dkb_sha256`, `ontology_build_id` so a re-audit after any change is unambiguous.

## 8. What QA explicitly checks that automation cannot
1. **H3 visual over-claim** — the only defect requiring the image (§4).
2. **Naturalness/pragmatics** — whether a technically valid caption reads oddly for a human (feeds template governance, doc 02 §7).
3. **Differential correctness in context** — whether an educational caption's "unlike X" clause is apt for the actual image, not just DKB-legal.
These three are the justification for human-in-the-loop review on top of an already strict automated battery.
