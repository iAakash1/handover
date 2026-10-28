# PlantVLM

Fine-tunes **Qwen2.5-VL-7B-Instruct** with LoRA on **150 manually validated captions**
covering **30 tomato leaf images** across **6 disease classes**.

The validated captions in `1stJuly/Revised_captions.xlsx` are the only supervision.
No template captions. No model-generated captions. No caption is rewritten — the
single transformation applied is whitespace normalization (restoring the space
after a full stop where two sentences were concatenated).

---

## Run it

```powershell
C:\Users\Admin\Desktop\VLM\venv\Scripts\Activate.ps1
cd C:\Users\Admin\Desktop\VLM\newpipe

python scripts/00_verify.py --gpu          # 28 checks, ~5 min  (already passing)
python scripts/02_train.py                 # THE RUN, ~15-25 min

python scripts/03_infer.py --split val --baseline   # control condition
python scripts/03_infer.py --split val              # fine-tuned
python scripts/04_evaluate.py --split val --compare-baseline
```

`data/` is already built. Rebuild it any time with `python scripts/01_build_dataset.py`.

Change anything without editing a file:

```powershell
python scripts/02_train.py --set train.epochs=12 --set lora.r=8
python scripts/01_build_dataset.py --set split.fold=2
```

---

## Architecture

```
        1stJuly/Revised_captions.xlsx            1stJuly/Dataset/<class>/N.JPG
                    │                                        │
                    └──────────────┬─────────────────────────┘
                                   ▼
  ┌───────────────────────────────────────────────────────────────────────┐
  │  DATA LAYER          plantvlm/data/                                   │
  │    sources.py   validated captions in   ◄── SWAP POINT: new dataset   │
  │    pairs.py     bind caption to a verified image file                 │
  │    verify.py    invariant GATE — refuses to pass bad data             │
  │    splits.py    group by image, stratify by class                     │
  └───────────────────────────────┬───────────────────────────────────────┘
                                  ▼
                    data/{pairs,train,val,test}.jsonl        ← the contract
                                  │
  ┌───────────────────────────────┴───────────────────────────────────────┐
  │  MODEL LAYER         plantvlm/models/                                 │
  │    base.py        the VLMAdapter contract                             │
  │    qwen2_5_vl.py  the only file that knows what Qwen is ◄── SWAP POINT│
  └───────────────────────────────┬───────────────────────────────────────┘
                                  ▼
  ┌───────────────────────────────────────────────────────────────────────┐
  │  TRAINING            plantvlm/training/                               │
  │    dataset.py  conversation formatting at runtime, via the adapter    │
  │    trainer.py  accumulation, validation, early stopping, checkpoints  │
  │    run.py      run isolation + lineage        ◄── REPRODUCIBILITY     │
  └───────────────────────────────┬───────────────────────────────────────┘
                                  ▼
             runs/run_NNN_foldK/  config.yaml · lineage.json · data/
                                  checkpoints/ · logs/ · predictions/ · evaluation/
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
      plantvlm/inference.py           plantvlm/evaluation.py
      base + fine-tuned generation    5-reference metrics, class retrieval
                                      with human ceiling, safety checks
```

A layer may only import from layers above it. Nothing outside `models/` imports a
model class; nothing outside `data/sources.py` knows the supervision is a
spreadsheet.

---

## The three decisions that shape this project

**Five captions per image become five training rows.** Not duplicates — five valid
answers to one question. Five gradient updates on the same pixels toward five
different targets mean no single string minimises the loss, so the model is pushed
toward the visual evidence instead of a memorised sentence. With 18 training
images this is the strongest regularisation available, and it comes free from
annotation work already done.

**No augmentation.** The captions make absolute spatial claims ("along the right
margin", "near the upper left section", "the tip appears darker"). A flip makes the
caption false; a crop can delete the feature being described. `train.augment` is
`false` and the dataset **raises** if it is switched on without a spatial-safe
transform.

**The vision tower is never adapted.** LoRA targets are built by walking the model
and excluding the vision encoder; the trainer asserts no `visual` parameter is
trainable and refuses to start otherwise. An image encoder cannot be learned from
18 photographs.

---

## Reading the evaluation

Every report prints **class retrieval accuracy together with its human ceiling** —
the same metric scored on the validated captions themselves.

The ceiling matters. By annotation guideline 5.3 the captions never name a disease,
and several classes share a visual vocabulary (Bacterial spot ↔ Early Blight,
Leaf Mold ↔ Mosaic Virus). Measured over all 60 held-out human-written captions,
this metric recovers the true class only ~40% of the time against 17% chance.
**A model near the ceiling is doing as well as this supervision allows.** Read the
score against the ceiling, never against 100%.

BLEU and ROUGE-L are computed against all five references per image. With six
held-out images they are a progress signal, not a result. Read the generated
captions yourself.

---

## Extending

**A new dataset** — add a `CaptionSource` subclass to `plantvlm/data/sources.py`,
register it in `SOURCES`, point `data.source.type` at it. Nothing else changes.

**A new VLM** — copy `plantvlm/models/qwen2_5_vl.py`, implement the same methods,
add it to `ADAPTERS`, set `model.adapter`. The trainer, dataset, inference engine
and evaluator are model-agnostic and need no edits.

**Cross-validation** — `--set split.fold=K` for K in 0..4 rotates which image of
each class is held out. Running all five folds tests every one of the 30 images
exactly once, and each run lands in its own directory, so nothing is overwritten.

---

## Known constraints

* **18 training images.** Five captions each multiply the rows, not the visual
  evidence. Expect good results on the held-out six and weak transfer to leaves
  photographed differently. The fix is more annotated images, not more captions
  per image.
* **The model will describe, never diagnose.** No validated caption contains a
  disease name, so the model cannot produce one.
* **7 captions are flagged** in `data/caption_defects.json` — 6 reference other
  images, 1 ends mid-sentence. Reported, never edited. If the annotator revises
  them, re-run `01_build_dataset.py`.
* **Peak VRAM measured at 19.4 GB** of 24 GB. On OOM, `--set train.micro_batch=1
  --set train.grad_accum=8` keeps the effective batch at 8.
