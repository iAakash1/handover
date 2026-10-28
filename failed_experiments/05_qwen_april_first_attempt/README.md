# 05 · First Qwen2.5-VL fine-tuning attempt (April 2026)

The first run of `projects/06_qwen25vl_plantvillage/qwen/02_finetune_qwen25vl.py` crashed, and the demo app would not
load. Instead of editing by hand each time, we wrote small patch scripts. These are those patches, kept so the
problems and their fixes are on record. **They were already applied; don't run them again.**

| Script | Problem | Fix it applied |
|---|---|---|
| `fix_finetune.py` | Captions cut at `MAX_LENGTH = 256`; processor truncation changed the number of image tokens, so Qwen's processor refused the batch; the collate function did not pad | MAX_LENGTH 512, no truncation, padding-aware `collate_fn` |
| `fix_clip.py` | CLIP weights would not load (`.bin` files blocked by the torch-version check in transformers) | load CLIP with `use_safetensors=True` |
| `fix_app2.py` | The app used `Qwen2VLForConditionalGeneration` and Flash-Attention 2, which isn't available on our Windows machine | `Qwen2_5_VLForConditionalGeneration`, plain bf16 attention |

After these fixes the run trained normally: **98.50%** accuracy (project 06).
