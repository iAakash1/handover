# LLaVA-1.6 and BLIP-2 LoRA adapters (24–25 Mar 2026)

**What is here:** `llava_plantvillage/best`: LLaVA-1.6-Mistral-7B LoRA (r64). `blip2_plantvillage/best`: BLIP-2 OPT-2.7B LoRA. Both were trained to write structured diagnoses for PlantVillage leaves.

**Results:** On the shared 200-image test: BLIP-2 **89.00%** (macro-F1 0.8928); LLaVA-1.6 2.00%.

**Finding:** BLIP-2 learned the format well. LLaVA answered with long generic descriptions and almost never named the disease (178 of 200 had no class), so the team moved to Qwen2.5-VL.
