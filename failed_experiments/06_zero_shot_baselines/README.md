# 06 · VLMs without fine-tuning (zero-shot)

Before and after each fine-tuning run we asked the untouched models the same question. They don't know plant
diseases:

| Model | Data | Accuracy | Report |
|---|---|---:|---|
| Qwen2.5-VL-7B-Instruct | PlantVillage, 200 validation images | 8.50% | `evaluation/qwen_zeroshot/` |
| Qwen2.5-VL-7B-Instruct | TIH tomato, 649 validation images | 8.47% (macro-F1 0.0766; 587 answers named no class) | `../08_tih_aborted_training_run/reports/validation_base/` |
| Florence-2-base-ft | PlantVillage, 200 validation images | 0.00% | `evaluation/florence_zeroshot/` |
| Seven VLMs, June benchmark | 30 expert-checked images | unreliable; Qwen3-VL-8B gave degenerate output on 16 of 30 | PoCRA report, `docs/reports/06-PoCRA-PCCOE-Tomato-Mango-Report-Version1.docx` |

The same Qwen2.5-VL went to 98.50% (PlantVillage) and 92.76% (TIH) after LoRA/QLoRA fine-tuning.

```powershell
cd evaluation
$env:MAX_EVAL = 5; ..\.venvs\vlm\Scripts\python vlm_eval.py qwen out_qwen          # base model, no adapter
..\.venvs\vlm\Scripts\python vlm_predict.py --model qwen_base --image <photo>
```
