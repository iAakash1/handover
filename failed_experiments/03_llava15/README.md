# 03 · LLaVA-1.5-7B + LoRA (March 2026): stopped by a batch-collation error

Our first LLaVA attempt. Training stopped with a batch-collation error, and no adapter
and no script survived (searched on this PC and in all our GitHub repositories, October 2026). It is recorded in the
PoCRA report (`docs/reports/06-PoCRA-PCCOE-Tomato-Mango-Report-Version1.docx`).

**Likely cause:** LLaVA cuts each photo into image patches and the text into tokens of different lengths. With
batch size > 1 the default collator tries to stack tensors of different shapes and fails. The rebuilt LLaVA-1.6 script uses
batch size 1 with gradient accumulation for this reason (`../02_llava16_mistral_lora/02_finetune_llava.py`).
