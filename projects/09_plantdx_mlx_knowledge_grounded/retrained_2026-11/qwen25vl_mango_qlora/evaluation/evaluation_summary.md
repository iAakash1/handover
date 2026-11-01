# Evaluation Summary: Base vs Fine-tuned (Qwen2.5-VL, mango)

- Samples evaluated: **200**
- Base accuracy: **0.180** | Fine-tuned accuracy: **0.860**
- Base macro-F1: **0.133** | Fine-tuned macro-F1: **0.891**

## Statistical significance (paired, base vs fine-tuned)
| Metric | Base mean | Fine-tuned mean | Difference | t-test p | Significant (p<0.05) |
|---|---|---|---|---|---|
| bleu1 | 0.061 | 0.324 | +0.264 | 1.133e-43 | yes |
| bleu2 | 0.023 | 0.245 | +0.222 | 6.12e-32 | yes |
| bleu3 | 0.004 | 0.197 | +0.193 | 1.013e-25 | yes |
| bleu4 | 0.001 | 0.162 | +0.161 | 2.823e-20 | yes |
| rouge_l | 0.098 | 0.457 | +0.359 | 3.351e-65 | yes |
| meteor | 0.162 | 0.471 | +0.309 | 3.718e-45 | yes |
| cider | 0.000 | 1.073 | +1.073 | 1.234e-16 | yes |
| bertscore_f1 | 0.843 | 0.911 | +0.068 | 7.578e-83 | yes |

See `metrics.json`, `per_disease.csv`, `hallucinations.csv`, and `figures/` for detail.
