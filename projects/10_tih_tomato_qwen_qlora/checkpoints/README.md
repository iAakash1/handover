# TIH tomato Qwen2.5-VL QLoRA adapters (Sep 2026)

**What is here:** `tih_tomato_qwen_qlora_strong` (15 Sep, LoRA r32, 392 px) and `tih_tomato_moreaccurate` (16 Sep, LoRA r64, 532 px) are trained on the TIH private tomato dataset (11 classes). Each folder also has its split manifest and run config.

**Results:** Test: strong 87.98%, moreaccurate **92.76%** (macro-F1 0.9378). Cross-domain (176 outside images): 69.32%.

**Finding:** Bigger LoRA rank and higher image resolution gave +4.8 points.
