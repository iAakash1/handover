# PlantCaption — a two-part caption-generation pipeline for plant-disease VLM training

This pipeline turns a small set of plant-disease photographs into a high-quality,
5-captions-per-image seed dataset for fine-tuning a vision-language model. It is
built to demonstrate **two kinds of understanding in every caption**:

| Part | Name | What it captures | Source |
|------|------|------------------|--------|
| **A** | Agricultural understanding | crop, disease, causal pathogen, severity, canonical symptoms, management | curated knowledge base, anchored to the ground-truth label |
| **B** | Image-pattern understanding | shot type, lighting (incl. dark conditions), a hand/finger/ruler/holder in frame, background, focus, wetness, % of leaf affected | a vision model that *looks at the pixels* |

Every one of the 5 captions per image weaves both parts together, so the dataset
teaches the downstream VLM to describe *both the agriculture and the photograph*.

---

## Why this design

* **5 captions per image** is the COCO standard and is what gives the trained
  model robust, non-repetitive language. The five captions use five different
  registers (a clinical report, a photograph description, a symptom walk-through,
  a field-scout note, and a tight integrated label) so they are genuinely
  diverse — this avoids the loss-collapse that identical captions cause.
* **The disease is taken from the label, not guessed by the vision model.** For
  new crops (cotton, sugarcane, maize) a zero-shot VLM will invent the wrong
  pathogen. Anchoring the diagnosis to the known label keeps the agronomy
  correct; the vision model is used only for the visual scene (Part B). This is
  the same label-grounded captioning strategy used in recent agricultural-VLM
  papers (e.g. Agro-Consensus, 2025), adapted to also capture the photographic
  attributes your brief asked for.

---

## Folder layout

```
VLM/                                  <- your project root
├── captioning/                       <- this package
│   ├── config.py                     edit paths / backend / settings here
│   ├── agronomy_kb.py                Part A: knowledge base + label parser
│   ├── visual_analyzer.py            Part B: Qwen2.5-VL / mock backends
│   ├── caption_synthesizer.py        fuses A + B into 5 captions
│   ├── dataset_builder.py            walks images, writes all outputs
│   ├── run_pipeline.py               command-line entry point
│   ├── review_app.py                 Streamlit review console (the demo)
│   ├── make_sample_images.py         generates synthetic test images
│   └── requirements.txt
├── images/                           <- put your 20 images here
└── outputs/                          <- generated artefacts land here
```

## How to organise your 20 images

Either layout works — the label parser handles both:

* **Sub-folders named after the class** (recommended):
  `images/Cotton___Bacterial_blight/photo1.jpg`
* **Flat files whose name encodes the class:**
  `images/cotton___bacterial_blight_01.jpg`

The parser understands `___` separators, hyphens, spaces, `corn_(maize)`,
camelCase like `YellowLeaf`, and common aliases (corn→maize, capsicum→bell
pepper, etc.). Run `python agronomy_kb.py` to see every class it knows and to
test your own label strings.

---

## Quick start

```bat
:: from VLM\captioning, with your venv active
..\venv\Scripts\activate
pip install -r requirements.txt

:: 0) (optional) make synthetic images to test the plumbing
python make_sample_images.py

:: 1) dry run, no model — confirms everything wires up (seconds)
python run_pipeline.py --backend mock

:: 2) real run on your 20 images with local Qwen2.5-VL
python run_pipeline.py --backend qwen

:: 3) review + edit + export the dataset
streamlit run review_app.py
```

---

## Outputs (written to `outputs/`)

* **`captions_master.json`** — one rich object per image: the full Part A and
  Part B fields plus all 5 captions. This is the human-review file.
* **`train.jsonl`** — one record per caption (20 images × 5 = 100 lines), in a
  `messages` format with an `<image>` token, ready for Qwen2.5-VL SFT. Each line
  also carries a `meta` block (crop, disease, pathogen, severity, archetype,
  scene attributes) for analysis and filtering.
* **`captions_review.csv`** — a flat spreadsheet of every caption with its
  structured fields; the easiest thing to hand your supervisor.
* **`visual_analysis_cache.json`** — caches the expensive Part-B output so
  re-runs (e.g. after tweaking caption wording) skip the vision model. Delete it
  or pass `--no-cache` to force a fresh visual pass.

---

## Extending the knowledge base

Add a new disease by dropping one entry into the `KB` dict in `agronomy_kb.py`
(key = `"crop|disease"`) and, if the label wording is unusual, a line or two in
`_CROP_ALIASES` / `_DISEASE_ALIASES`. Nothing else needs to change.

---

## Two-part design, made explicit for your review meeting

For any single image you can point to:

1. **Part A in `captions_master.json` → `agronomy`** — proves agricultural
   understanding (correct pathogen, symptom vocabulary, severity, management).
2. **Part B in `captions_master.json` → `visual`** — proves image-pattern
   understanding (it detected the hand, the lighting, the framing, the %
   affected).
3. **The 5 captions** — show the two are fused into natural, diverse language
   suitable for training.
