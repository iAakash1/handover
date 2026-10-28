# BLIP-2 re-run with full evaluation (15 Jun 2026)

**What is here:** A 1-epoch BLIP-2 OPT-2.7B LoRA re-run with BLEU/ROUGE/CLIP evaluation. Model: `outputs/blip2_plantvillage/best`.

**Results:** Its own June script recorded BLEU-1 0.2081 and 0% classification (`evaluation_report.json`); re-evaluated with the April protocol: 89.00% accuracy, macro-F1 0.8948 (`evaluation_reports_oct2026/blip2_june`).

**Finding:** The 0% was a scoring issue in the June script, not the model; the longer March run reached 89.00%.
