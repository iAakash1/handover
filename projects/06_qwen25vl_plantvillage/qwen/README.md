# Qwen2.5-VL-7B LoRA (6–7 Apr 2026): the model that worked

**What is here:** `01_prepare_dataset.py` writes structured captions (`captions.json`, 90/10 split, seed 42); `02_finetune_qwen25vl.py` trains LoRA r32 for 3 epochs; `03_inference.py` / `04_evaluate.py` test it; `05_app.py` is the Streamlit app. Model: `outputs/qwen25vl_plantvillage/best`.

**Results:** **98.50%** accuracy, macro-F1 0.9827, BLEU-4 0.4556, severity R² 0.9654 (`evaluation_report.json`). Without fine-tuning the same model scores 8.50%.

**Finding:** Fine-tuning, not the base model, is what makes the diagnosis reliable.
