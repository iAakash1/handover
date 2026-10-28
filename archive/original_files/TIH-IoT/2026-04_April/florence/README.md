# Florence-2-base-ft attempt (3 Apr and 15 Jun 2026)

**What is here:** Fine-tuning scripts plus `patch_florence2.py`, which tried to fix incompatibilities with transformers 5.3.

**Results:** Fine-tuning never finished. Zero-shot (re-run with transformers 4.46.3): 0.00%; it describes "a green leaf on the ground" without naming any disease (`evaluation_reports_oct2026/florence_zeroshot`).

**Finding:** Dropped: too small and too generic for diagnosis without working fine-tuning.
