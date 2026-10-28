# fix_finetune.py
# Run this once: python fix_finetune.py
# It rewrites 02_finetune_qwen25vl.py with the correct Dataset and collate_fn

with open("02_finetune_qwen25vl.py") as f:
    txt = f.read()

# ── 1. Bump MAX_LENGTH to 512 ─────────────────────────────────────────────
txt = txt.replace("MAX_LENGTH   = 256", "MAX_LENGTH   = 512")

# ── 2. Remove truncation/padding from processor call ─────────────────────
# The Qwen2.5-VL processor validates image token counts against the text,
# so truncation must not be applied at the processor level.
OLD_ENC = """        enc_full = self.processor(
            text=[text_full],
            images=[image],
            return_tensors="pt",
            padding="max_length",
            max_length=MAX_LENGTH,
            truncation=True,
        )"""

NEW_ENC = """        enc_full = self.processor(
            text=[text_full],
            images=[image],
            return_tensors="pt",
        )"""

txt = txt.replace(OLD_ENC, NEW_ENC)

# ── 3. Remove truncation from prompt-only tokenizer call ──────────────────
OLD_PROMPT = """        enc_prompt = self.processor.tokenizer(
            text_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
        )"""

NEW_PROMPT = """        enc_prompt = self.processor.tokenizer(
            text_prompt,
            return_tensors="pt",
        )"""

txt = txt.replace(OLD_PROMPT, NEW_PROMPT)

# ── 4. Replace collate_fn with padding-aware version ─────────────────────
# Without fixed-length truncation, sequences have variable length per batch.
# We pad input_ids and labels to the longest sequence in each batch.

OLD_COLLATE = """def collate_fn(batch):
    \"\"\"Stacks fixed-size tensors; concatenates pixel_values (variable patches).\"\"\"
    return {
        "input_ids":      torch.stack([b["input_ids"]      for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "pixel_values":   torch.cat( [b["pixel_values"]    for b in batch], dim=0),
        "labels":         torch.stack([b["labels"]          for b in batch]),
    }"""

NEW_COLLATE = """def collate_fn(batch):
    \"\"\"Pads variable-length sequences to the longest in the batch.\"\"\"
    import torch.nn.functional as F
    max_len = max(b["input_ids"].shape[0] for b in batch)
    pad_id  = 0  # Qwen2.5-VL pad token id

    input_ids_list      = []
    attention_mask_list = []
    labels_list         = []

    for b in batch:
        seq_len = b["input_ids"].shape[0]
        pad_len = max_len - seq_len
        input_ids_list.append(
            F.pad(b["input_ids"],      (0, pad_len), value=pad_id)
        )
        attention_mask_list.append(
            F.pad(b["attention_mask"], (0, pad_len), value=0)
        )
        labels_list.append(
            F.pad(b["labels"],         (0, pad_len), value=-100)
        )

    return {
        "input_ids":      torch.stack(input_ids_list),
        "attention_mask": torch.stack(attention_mask_list),
        "pixel_values":   torch.cat([b["pixel_values"] for b in batch], dim=0),
        "labels":         torch.stack(labels_list),
    }"""

txt = txt.replace(OLD_COLLATE, NEW_COLLATE)

with open("02_finetune_qwen25vl.py", "w") as f:
    f.write(txt)

# ── Verify all 4 replacements landed ──────────────────────────────────────
checks = [
    ("MAX_LENGTH   = 512",       "MAX_LENGTH bumped to 512"),
    ("return_tensors=\"pt\",\n        )\n",  "Truncation removed from processor"),
    ("F.pad(b[\"input_ids\"]",   "Padding-aware collate_fn applied"),
]
print("Patch results:")
for needle, label in checks:
    found = needle in txt
    print(f"  {'OK' if found else 'MISS'}  {label}")

print("\nDone. Run: python 02_finetune_qwen25vl.py")
