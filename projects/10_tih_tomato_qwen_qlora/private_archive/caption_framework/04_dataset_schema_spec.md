# 04 — Dataset Schema Specification (Task 6)

**Deliverable 5 of 8.** Defines the **canonical caption record** (the single source of truth for Stage 3), the **grounded instruction design** (the user-turn side of each training example), the **split policy**, and the **per-model format converters** for Qwen2.5-VL, Qwen3-VL, InternVL3, Gemma-3, and MLX (mlx-vlm). Includes the recommendation on one-canonical-schema-vs-many and its justification.

Headline recommendation (Task 6): **Maintain ONE canonical, tool-agnostic JSON schema and convert to each trainer's format with deterministic adapters at Stage 3.** Do **not** generate separate datasets per model. Justification in §7.

---

## 1. The canonical caption record (`captions/caption_library.jsonl`, one JSON object per line)

```json
{
  "schema_version": "1.0",
  "caption_id": "cap_9f2a…",                     // = SHA256(image_id ‖ caption_seed ‖ template_id)[:16]
  "image": {
    "id": "PlantVillage/Tomato___Early_blight/0a1b2c.JPG",
    "path": "tomato/PlantVillage/Tomato___Early_blight/0a1b2c.JPG",
    "dataset": "PlantVillage",                   // PlantVillage | MangoLeafBD
    "crop": "tomato"                             // tomato | mango
  },
  "label": {
    "disease_id": "tomato_early_blight",         // FK into dkb.json / caption_ontology.json
    "class_label": "Early Blight",
    "is_pathogen_disease": true,
    "agent_category": "fungus"
  },
  "instruction": {
    "task_type": "describe",                     // §4
    "template_id": "I-DESC-03",
    "text": "Describe the condition of this leaf."
  },
  "response": {
    "text": "This tomato leaf shows early blight. Scattered brown concentric-ringed lesions are visible on the lower leaves, with a surrounding yellow halo.",
    "style": "two_sentence",
    "length_band": "medium",
    "register": "descriptive",
    "hedged": false,
    "token_count": 27
  },
  "concepts": ["disease_identity","primary_sign","extent","leaf_location","chlorosis"],
  "provenance": {
    "global_seed": 20260711,
    "base_seed": "…", "caption_seed": "…",
    "template_id": "T-TS-01",
    "instruction_template_id": "I-DESC-03",
    "expansion_edges": [{"type":"ADD_color","value":"brown","source_field":"color_vocabulary"}],
    "vocab_choices": [{"slot":"PRIMARY","surface":"brown concentric-ringed lesions","source_field":"diagnostic_visual_features"}],
    "validator_report": { "verdict":"accept","attempts":1,"validators":{"V1":{"pass":true},"…":"…"} },
    "fallback": false,
    "dkb_sha256": "…", "ontology_build_id": "onto_2026…",
    "template_set_version": "1.0", "vocabulary_version": "1.0",
    "config_hash": "…", "generator_version": "1.0",
    "created_utc": "2026-07-11T00:00:00Z"
  },
  "split": "train",                              // train | val | test | diagnostic  (§5)
  "qa": { "reviewed": false, "verdict": null, "reviewer_id": null, "notes": null }
}
```

Rules:
- The canonical record is **model-agnostic**: no `<image>` tokens, no chat special tokens, no role scaffolding. Those are introduced only by converters (§6).
- `image.path` is relative to `experiments/` so paths are portable.
- `provenance` is complete enough to regenerate `response.text` bit-for-bit (doc 00 §6). It is retained in the canonical library and **stripped** by converters (training files carry only what the trainer needs) but retained in `metadata/provenance/`.
- One image yields multiple records (multiple captions × instruction pairings), all sharing `image.id` and therefore the same `split` (§5).

## 2. Label mapping (`metadata/label_map.json`)
Folder names → `disease_id`. Authored once, reviewed, immutable. Examples:
```json
{
  "PlantVillage": {
    "Tomato___healthy":"tomato_healthy",
    "Tomato___Bacterial_spot":"tomato_bacterial_spot",
    "Tomato___Early_blight":"tomato_early_blight",
    "Tomato___Late_blight":"tomato_late_blight",
    "Tomato___Leaf_Mold":"tomato_leaf_mold",
    "Tomato___Septoria_leaf_spot":"tomato_septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite":"tomato_spider_mites",
    "Tomato___Target_Spot":"tomato_target_spot",
    "Tomato___Tomato_mosaic_virus":"tomato_mosaic_virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus":"tomato_yellow_leaf_curl_virus"
  },
  "MangoLeafBD": {
    "Healthy":"mango_healthy","Anthracnose":"mango_anthracnose","Bacterial Canker":"mango_bacterial_canker",
    "Cutting Weevil":"mango_cutting_weevil","Die Back":"mango_die_back","Gall Midge":"mango_gall_midge",
    "Powdery Mildew":"mango_powdery_mildew","Sooty Mould":"mango_sooty_mould"
  }
}
```
> The exact PlantVillage folder strings must be reconciled against the actual directory names in `tomato/PlantVillage/` at build time (PlantVillage spelling/spacing varies by mirror). The implementer confirms each folder resolves to exactly one `disease_id`; an unmapped folder is a hard error. This is the only place folder strings are hard-coded.

## 3. Instruction-tuning example = (instruction, image, response)
Instruction tuning needs a **user turn**. The caption is the **assistant response**. Both are generated by this framework and both are grounded — the response's allowed concepts are a function of the instruction's `task_type` (a "what color?" question must be answered only from the `lesion_color` concept). This is enforced by pairing each instruction task type with a **response constraint** (§4.2), fed into the concept selector as an additional filter.

## 4. Instruction bank (`templates/instructions.json`)

### 4.1 Task types
| task_type | Intent | Example user turns (templated, ≥6 each) | Response = |
|-----------|--------|----------------------------------------|-----------|
| `describe` | open description | "Describe the condition of this leaf." / "What can you see on this leaf?" | full caption (any style) |
| `identify` | name the condition | "What disease or condition affects this leaf?" / "Identify the problem shown." | disease identity + ≥1 primary sign (short/single) |
| `signs` | enumerate visible signs | "What symptoms are visible on this leaf?" | primary_sign (+ selected descriptive concepts), no bare disease name required |
| `color_qa` | targeted attribute | "What color are the lesions on this leaf?" | `lesion_color` only (+ head noun) |
| `location_qa` | targeted attribute | "Where on the leaf do the symptoms appear?" | `leaf_location`/`lesion_distribution` only |
| `crop_qa` | host | "What crop is this leaf from?" | `host` only |
| `differential` | discrimination | "How can you tell this apart from similar diseases?" | educational template with `differential` (only for classes with `confused_with`) |
| `healthy_check` | binary health | "Is this leaf healthy or diseased?" | health verdict + (if diseased) disease identity |

### 4.2 Response constraints (fed to the concept selector as a mask)
- `color_qa` ⇒ `required={lesion_color}`, `allowed={lesion_color, host}`, force `short/single` style; if the disease has no `lesion_color` concept (e.g., healthy, ToMV, cutting_weevil), the pairing is **invalid** and skipped.
- `location_qa` ⇒ `required={leaf_location|lesion_distribution}`.
- `crop_qa` ⇒ `required={host}`; response is a single short clause.
- `identify`/`healthy_check` ⇒ `required={disease_identity}` (+ primary_sign for identify).
- `differential` ⇒ `required={disease_identity, primary_sign, differential}`; only when `confused_with` non-empty.
- `describe`/`signs` ⇒ standard budget (doc 00 §4).
Invalid pairings (task requires a concept the disease lacks) are pruned before generation; the budget planner (doc 00 §7.4) samples only valid `(task_type, style)` combos per disease.

### 4.3 Instruction diversity
Instruction texts are themselves templated (≥6 paraphrases per task type) and selected with the same anti-domination + seeded sampling as captions (doc 02 §5), so the *user* turn is diverse too. Instruction paraphrases are generic English (no domain content) and live in a reviewed closed set.

## 5. Split policy (`datasets/splits/`)
- **Grouped by image.** All records of one `image.id` go to the same split (prevents caption-level leakage of an image across train/test).
- **Stratified** by `disease_id` (and thereby crop) to preserve class proportions in each split.
- **Ratios:** `train 0.80 / val 0.10 / test 0.10` by *image*, seeded and recorded as `datasets/splits/{train,val,test}_image_ids.txt`.
- **Diagnostic split (held-out, small):** a curated `diagnostic` set of images from the **hardest confusable pairs** per the DKB `confused_with` (e.g., early-blight vs target-spot vs Septoria; anthracnose vs bacterial-canker; powdery-mildew vs sooty-mould). Used in Stage 5 to probe discrimination, not for training. Drawn from the test pool so it never overlaps train/val.
- Splits are fixed and versioned so all four models train/evaluate on identical partitions (fair comparison, §7).

## 6. Per-model converters (Stage 3 adapters; `datasets/<model>/`)
Each converter is a **pure, deterministic function** `canonical_record → trainer_line`. They add only role scaffolding and image placeholders; they never alter `response.text`. Converters read `split` and emit `{train,val}.jsonl` per model (test/diagnostic reserved for eval).

### 6.1 Qwen2.5-VL and Qwen3-VL (`datasets/qwen2_5_vl/`, `datasets/qwen3_vl/`)
Same family; identical data schema (Qwen3-VL differs in model internals, not data-prep format). Messages with typed content list (consumed by Qwen chat template / ms-swift / LLaMA-Factory, which insert `<|vision_start|><|image_pad|><|vision_end|>` automatically):
```json
{"messages":[
  {"role":"user","content":[
    {"type":"image","image":"tomato/PlantVillage/Tomato___Early_blight/0a1b2c.JPG"},
    {"type":"text","text":"Describe the condition of this leaf."}]},
  {"role":"assistant","content":[{"type":"text","text":"<response.text>"}]}
]}
```

### 6.2 InternVL3 (`datasets/internvl3/`)
LLaVA-style `conversations` schema with the `<image>` placeholder (InternVL / ms-swift convention):
```json
{"id":"cap_9f2a…",
 "image":"tomato/PlantVillage/Tomato___Early_blight/0a1b2c.JPG",
 "conversations":[
   {"from":"human","value":"<image>\nDescribe the condition of this leaf."},
   {"from":"gpt","value":"<response.text>"}]}
```

### 6.3 Gemma-3 (`datasets/gemma3/`)
Messages/content-list form; the Gemma-3 processor chat template inserts `<start_of_turn>`/`<end_of_turn>` and the `<start_of_image>` token:
```json
{"messages":[
  {"role":"user","content":[
    {"type":"image","image":"…/0a1b2c.JPG"},
    {"type":"text","text":"Describe the condition of this leaf."}]},
  {"role":"assistant","content":[{"type":"text","text":"<response.text>"}]}
]}
```
(Structurally identical to Qwen's content-list form; the *chat template*, applied by the model's processor at train time, differs — which is exactly why we keep one canonical schema and let each processor render its own tokens.)

### 6.4 MLX / mlx-vlm (`datasets/mlx_vlm/`) — the actual fine-tuning tool on M4 Pro
`mlx_vlm.lora` consumes a chat dataset with an image field and a `messages` list, then applies the target model's own chat template. Emit the same messages/content-list form as §6.1 with a top-level `images` array:
```json
{"images":["tomato/PlantVillage/Tomato___Early_blight/0a1b2c.JPG"],
 "messages":[
   {"role":"user","content":"<image>Describe the condition of this leaf."},
   {"role":"assistant","content":"<response.text>"}]}
```
> **Version caveat (must verify at implementation):** mlx-vlm's expected dataset schema has changed across releases (some versions want `messages`+`images`, others a HF-dataset with `image`+`conversations`). The converter targets the **installed** mlx-vlm version; pin it in `datasets/mlx_vlm/README.md` and validate one sample end-to-end with `mlx_vlm.lora --help`/a 1-step dry run before bulk conversion. Because the canonical schema carries everything, only this thin adapter changes if mlx-vlm's format shifts.

### 6.5 Converter contract
- Deterministic; no randomness; no text edits to `response`/`instruction`.
- Validates each emitted line against the trainer's minimal schema (image path exists, roles alternate user→assistant, exactly one image per example unless multi-image is intended).
- Emits a `datasets/<model>/manifest.json`: source `library_version`, split, record count, converter version, target-tool version.
- Idempotent: re-running reproduces identical files.

## 7. Recommendation & justification (Task 6 core question)
**Adopt one canonical schema (§1) + deterministic adapters (§6). Do not maintain separate per-model datasets.**

| Criterion | Single canonical + adapters (recommended) | Separate per-model datasets |
|-----------|-------------------------------------------|-----------------------------|
| Single source of truth | ✅ one library; adapters are derivations | ❌ 4–5 copies drift |
| Fair model comparison | ✅ identical content/splits guaranteed | ❌ hard to keep identical |
| Reproducibility | ✅ provenance in one place; adapters pure | ❌ provenance duplicated/lost |
| Adding a 5th model | ✅ write one adapter | ❌ regenerate a dataset |
| Trainer-format churn (esp. mlx-vlm) | ✅ change one thin adapter | ❌ regenerate everything |
| QA effort | ✅ review the canonical library once | ❌ review N datasets |
| Storage | ✅ captions once + light adapters | ❌ N full copies |

Justification narrative: the scientific content (the caption) is invariant across trainers; only *serialization* differs. Coupling content to serialization would fork the dataset and reintroduce drift — the very failure mode the DKB-as-single-source-of-truth design exists to prevent. Keeping one canonical library also guarantees that the four fine-tuned models (Qwen2.5-VL, Qwen3-VL, InternVL3, Gemma-3) are trained on **identical** examples and **identical** image-level splits, which is a precondition for a clean zero-shot-vs-fine-tuned and cross-model comparison in Stage 5 / the paper. Adapters are cheap, pure, and independently testable; the mlx-vlm version caveat (§6.4) is isolated to a single 30-line function instead of contaminating the dataset.
