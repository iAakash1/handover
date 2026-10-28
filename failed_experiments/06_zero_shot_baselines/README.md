# 06 · VLMs without fine-tuning (zero-shot)

Before and after each fine-tuning run we asked the untouched models the same question. They don't know plant
diseases:

| Model | Data | Accuracy | Report |
|---|---|---:|---|
| Qwen2.5-VL-7B-Instruct | PlantVillage, 200 validation images | 8.50% | `evaluation/qwen_zeroshot/` |
| Qwen2.5-VL-7B-Instruct | TIH tomato, 649 validation images | 8.47% (macro-F1 0.0766; 587 answers named no class) | `../08_tih_aborted_training_run/reports/validation_base/` |
| Florence-2-base-ft | PlantVillage, 200 validation images | 0.00% | `evaluation/florence_zeroshot/` |
| Four VLMs on MLX (Mac), 25 Jun 2026 | 16 leaf images (tomato, potato, pepper, cotton) | exact disease match 0–6% per model; right disease mentioned anywhere 19–50% | `projects/09_plantdx_mlx_knowledge_grounded/earlier_experiments/03_zero_shot_4vlm_benchmark_2026-06-25/report/` |
| Seven VLMs on MLX (Mac), 1 Jul 2026 | 30 expert-checked tomato images | best accuracy 23% (InternVL3-8B); Qwen3-VL-8B gave degenerate output on 16 of 30 | `projects/09_plantdx_mlx_knowledge_grounded/earlier_experiments/04_zero_shot_7vlm_benchmark_2026-07-01/report/`, PoCRA report `docs/reports/06-PoCRA-PCCOE-Tomato-Mango-Report-Version1.docx` |
| Qwen2.5-VL-7B-Instruct-4bit (MLX) | PlantVillage tomato 910 / MangoLeafBD 200 test images | 7.3% / 17.5% | `projects/09_plantdx_mlx_knowledge_grounded/reports/qwen25vl_<crop>_qlora/evaluation/` (base rows) |

The same Qwen2.5-VL went to 98.50% (PlantVillage) and 92.76% (TIH) after LoRA/QLoRA fine-tuning.

```powershell
cd evaluation
$env:MAX_EVAL = 5; ..\.venvs\vlm\Scripts\python vlm_eval.py qwen out_qwen          # base model, no adapter
..\.venvs\vlm\Scripts\python vlm_predict.py --model qwen_base --image <photo>
```

The two MLX benchmarks (code, images, every raw answer and both reports) ran on the Mac and are kept with the
other Mac work in `projects/09_plantdx_mlx_knowledge_grounded/earlier_experiments/`, which also has their run
commands. They need macOS on Apple Silicon.
