# Evaluation Summary: Base vs Fine-tuned (Qwen2.5-VL, tomato)

- Samples evaluated: **910**
- Base accuracy: **0.073** | Fine-tuned accuracy: **0.869**
- Base macro-F1: **0.034** | Fine-tuned macro-F1: **0.847**

## Statistical significance (paired, base vs fine-tuned)
| Metric | Base mean | Fine-tuned mean | Difference | t-test p | Significant (p<0.05) |
|---|---|---|---|---|---|
| bleu1 | 0.058 | 0.256 | +0.198 | 1.034e-129 | yes |
| bleu2 | 0.019 | 0.194 | +0.175 | 1.03e-108 | yes |
| bleu3 | 0.003 | 0.149 | +0.147 | 3.377e-82 | yes |
| bleu4 | 0.001 | 0.115 | +0.115 | 2.055e-56 | yes |
| rouge_l | 0.093 | 0.399 | +0.306 | 1.315e-234 | yes |
| meteor | 0.149 | 0.387 | +0.239 | 2.014e-142 | yes |
| cider | 0.000 | 0.817 | +0.816 | 1.555e-47 | yes |
| bertscore_f1 | 0.843 | 0.895 | +0.052 | 8.532e-194 | yes |

See `metrics.json`, `per_disease.csv`, `hallucinations.csv`, and `figures/` for detail.
