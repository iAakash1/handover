# VLM Teaching Session — Run of Show (classroom edition)

**Deck:** `VLM_Teaching_Session.pptx` · **28 slides** · **~22–25 min** + Q&A
**Presenters:** Harish · Aakash · Kamaksha
**Audience:** third-year engineering students, no prior VLM knowledge

> **Design rules this deck follows**
> 1. **Teach the concept first, then show our implementation.** Never the reverse.
> 2. **One core idea per slide.** Many slides are just a diagram or an image — you carry the rest.
> 3. **Participation every 3–5 slides.** Five purple *PAUSE* slides force the room to think.
> 4. **One continuous example.** The same late-blight leaf appears on slides 2 → 13 → 15 → 21,
>    so students follow *one image* through dataset → benchmark → captions → training → inference.
> 5. **Every number is real.** Nothing on any slide is invented or rounded up.

Full notes are embedded in the PPTX — each slide has **🎯 Objective · 🗣️ What to say ·
🚫 What NOT to say · ❓ Likely questions · ➡️ Transition**. Use Presenter View.

---

## Timing map

| Section | Slides | Time | Running |
|---|---|---:|---:|
| 0 · Opening hook | 1–2 | 1:30 | 1:30 |
| 1 · Foundations | 3–6 (+**quiz 5**) | 5:00 | 6:30 |
| 2 · Our data | 7–9 | 2:30 | 9:00 |
| **3 · Benchmarking (core)** | 10–18 (+**pause 11, 16**) | **8:00** | **17:00** |
| 4 · Teaching the model | 19–24 (+**pause 19**) | 5:00 | 22:00 |
| 5 · System | 25–26 | 1:30 | 23:30 |
| 6 · Recap + close | 27–28 | 1:30 | 25:00 |

**Need to hit 20 minutes?** Cut slide 3 (evolution) and slide 23 (five captions) → 26 slides.
**Never cut:** 13, 15, 17, 22, or any purple PAUSE slide.

---

## The five participation points

| # | Slide | Type | What you do | Time |
|---|---|---|---|---:|
| 1 | **5** | Quick quiz | "CNN, LLM or VLM?" × 3 scenarios — hands up | 45s |
| 2 | **11** | Think–pair | "What would you keep identical? What would you measure?" | 75s |
| 3 | **16** | **Prediction** ⭐ | "How many of 16 will the *best* model get?" — hands for each band | 60s |
| 4 | **19** | Discussion | "Where should captions come from? What goes wrong with each?" | 75s |
| 5 | **27** | Recap quiz | Three retrieval questions | 60s |
| — | 26 | *(built-in)* | Cover the right half, ask "what's wrong with this answer?", then reveal | 20s |

> **#3 is the most important teaching device in the deck.** Students who *commit to a guess*
> remember the result far better. Most rooms guess 8–15; the answer is 3. Do not leak it with
> your tone.

---

## Slide-by-slide

### Opening (1–2) · 1:30
**1 Title** — 30s. Introduce all three. Warn them it's interactive.
**2 Hook** — 60s. The hero leaf. *Ask and wait.* Take 2–3 answers, correct none.

### Section 1 · Foundations (3–6) · 5:00
**3 Evolution** — 60s. Six boxes, ~10s each. Land: *a VLM is an LLM that can also see.*
**4 CNN vs LLM vs VLM** ⭐ — 90s. Your best teaching slide. Walk the input→output row.
**5 ⚡ QUIZ** — 45s. Answers: ① CNN ② VLM ③ LLM.
**6 How a VLM works** — 75s. Eye → translator → mouth. Name the projector. Mention applications verbally.

### Section 2 · Our data (7–9) · 2:30
**7 Dataset samples** — 60s. **Real images.** Ask "which is healthy?" then "can you tell early
from late blight?" — most can't. That's the point.
**8 Class distribution** — 45s. 5,357 vs 373 = **14× imbalance**. Ask which class the model will
be worst at. A lesson that transfers to any ML project.
**9 Our goal** — 45s. Describe, don't classify. Introduce the **single-leaf constraint**.

### Section 3 · Benchmarking (10–18) · 8:00 ⭐ *the heart*
**10 Why benchmark** — 60s. Ask "how would you pick a model?" Then dismantle leaderboards.
**11 ⚡ THINK–PAIR** — 75s. Let them design the experiment. Let the room get noisy.
**12 Method** — 60s. Connect to their answers: *"several of you said keep the prompt identical — exactly."*
**13 The prompt** — 60s. Read the three gold lines slowly. Say **"remember these."**
**14 The models** — 60s. Eight models, two setups. Every chart = Setup A.
**15 One leaf, four models** ⭐⭐ — 100s. **The centrepiece.** Same leaf as slide 2. Read the four
species answers across: *Uncertain · a legume · **Soybean** · Unknown.* Call back to the prompt.
Killer line: **not one said late blight.**
**16 ⚡ PREDICTION** — 60s. Hands up per band. Count out loud.
**17 The result** — 90s. **Let the chart sit in silence for 3 seconds.** Then: *none were reliable,
best 3/16.* Compare to their guesses. **Do not rescue it.**
**18 Choosing Qwen** — 75s. Pre-empt the objection: *"you're thinking Gemma got 3 and Qwen got 0 —
correct."* Then: select for **trainability**, not score.

### Section 4 · Teaching the model (19–24) · 5:00
**19 ⚡ DISCUSSION** — 75s. When someone says "ask a model", hit them with: *"which model? The one
that got 3 out of 16?"* Let the silence land.
**20 Why fine-tune** — 45s. Bright graduate who's read everything, practised nothing.
**21 Flashcards** — 60s. **Same hero leaf on the front.** Say *"one word at a time"* twice.
**22 Where captions came from** ⭐ — 90s. **The credibility slide.** Read the **minus lines out loud**.
**23 Five captions** — 45s. Ask "which is correct?" → all five.
**24 LoRA** — 60s. Sticky-notes metaphor, twice. Four real stat boxes.

### Section 5 · System (25–26) · 1:30
**25 Live system** — 60s. Real screenshots. Explain confidence honestly.
**26 Before / after** — 30s. *Cover the right half first, ask, then reveal.* Then the honesty line:
**behaviour change, not proof of accuracy.**

### Section 6 · Close (27–28) · 1:30
**27 ⚡ RECAP** — 60s. ① encoder→projector→LLM ② circularity ③ ~0.5%.
**28 Status + close** — 30s. **Read the red column as confidently as the green.**

---

## The continuous example — one leaf, five appearances

| Slide | The same late-blight leaf appears as… |
|---|---|
| **2** | the hook — *"a farmer sends you this photo"* |
| **7** | one of the ten real classes in our dataset |
| **13** | the image the shared prompt was applied to |
| **15** | the image all **four** models got wrong |
| **21** | the front of the training flashcard |

Say this out loud at slide 15: *"this is the same leaf I showed you in the first minute."*
It is what turns a lecture into a story.

---

## Visual inventory — provenance of everything on screen

| Slide | Visual | Source |
|---|---|---|
| 2, 13, 15, 21 | Hero late-blight leaf | `crop_vlm_benchmark/images/…RS_LB 4857.JPG` — real benchmark image |
| 7 | 8-class sample grid | **Generated** from `datasets/tomato/processed/` — real images |
| 8 | Class distribution | **Generated** from real folder counts (18,160 images) |
| 17 | Disease-accuracy chart | **Generated** from `json_results/all_raw_results.json` |
| 18 | Format + latency charts | **Generated** from the same raw results |
| 15 | Four model verdicts | Verbatim from our recorded benchmark outputs |
| 23 | Early-blight leaf | **Preserved** from the previous deck |
| 25 | 3 × Streamlit screenshots | **Preserved** from the previous deck — our own app |
| 26 | Base vs fine-tuned quotes | Verbatim from our own runs |

**Every screenshot from the earlier deck is preserved.** Nothing was downloaded from the internet.

---

## Numbers you must be able to defend

| Claim | Value | Source |
|---|---|---|
| Tomato dataset | **18,160 images · 10 classes** | folder counts |
| Class imbalance | 5,357 (curl virus) vs 373 (mosaic) = **14×** | folder counts |
| Mango dataset | 4,000 · 8 classes · 500 each | folder counts |
| Models benchmarked | **8** (4 Setup A + 5 Setup B, Qwen in both) | `config.py` + second pipeline |
| Benchmark scale | 16 images · 6 conditions · 64 runs · 0 failures · ~22 min | `all_raw_results.json` |
| Best zero-shot | **3 / 16** (Gemma-3) | computed from recorded outputs |
| Qwen zero-shot | **0 / 16** | same |
| Qwen format adherence | **16 / 16** (Gemma-3 & Llama: 0/16) | same |
| Qwen latency | **5.3 s** median (Llama 50.8 s) | same |
| Trainable params | **0.484%** · 173 MB adapter | training artifacts |
| Peak memory | 19.4 / 24 GB | training run |
| Hand-written captions | 30 images · 150 captions · 5 per image | `Revised_captions.xlsx` |

**Never say:** "state of the art", "98%", "excellent accuracy", or any figure not in this table.

⚠️ **We have no fruit images and no whitefly / aphid / fruit-borer classes.** Our data is
**leaves only**, 10 tomato classes + 8 mango. If asked, say so plainly.

---

## Three sentences to rehearse until automatic

1. *"None of the eight models diagnosed reliably — the best got 3 of 16."*
2. *"We chose Qwen for format stability and speed, not accuracy."*
3. *"That's a fair criticism, and here's what we'd do about it."*

---

## Hardest four questions

**"Isn't a caption built from labels just a classifier in disguise?"**
> Largely fair. It describes the disease, not this specific leaf — which is why our hardest class
> performs poorly. The fix is a hybrid: reliable factual spine + one human line about this leaf.

**"Why not use GPT-4o to write the captions?"**
> Circularity. Our benchmark measured the best model at 3/16 — you'd bake its errors into the
> supervision.

**"16 images is too small to conclude anything."**
> For a precise *ranking*, correct, and we say so. But 0–3 out of 16 across four models is a clear
> signal, and it matched what we saw later on much larger test sets.

**"Why not just use a CNN?"**
> A CNN gives a label with no reasons. But honestly — **we never trained a CNN baseline**, so we
> can't claim we beat one. Our claim is about the *form* of the output.

---

## Pre-flight checklist

- [ ] Open the deck on the presenting machine — check Calibri renders
- [ ] Presenter View on, so you can see the 5-part notes
- [ ] Rehearse the two hand-overs (after slide 9, after slide 24)
- [ ] Time slides 10–18 — this section always runs long
- [ ] Practise slide 17 **without softening it**
- [ ] Practise the slide-26 reveal (cover right half → ask → uncover)
- [ ] Agree who fields concept vs implementation vs limitation questions

---

## Deliberately excluded

Not on any slide; raise only if asked: Disease Knowledge Base, ontology, vocabulary builder,
caption framework internals, validation engines, repository architecture, config systems,
determinism/content hashing, two-stage evaluation split.

**Why:** each requires explaining a *different* thing first, and none teaches what a VLM is.
One-sentence answer if asked: *"We built a cited reference the pipeline reads from, so captions
trace back to a source — happy to walk through it afterwards."*
