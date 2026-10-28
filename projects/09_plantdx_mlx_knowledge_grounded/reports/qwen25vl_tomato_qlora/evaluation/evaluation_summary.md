# Evaluation Summary: Base vs Fine-tuned (Qwen2.5-VL, tomato)

- Samples evaluated: **910**
- Base accuracy: **0.073** | Fine-tuned accuracy: **0.937**
- Base macro-F1: **0.034** | Fine-tuned macro-F1: **0.919**

## Statistical significance (paired, base vs fine-tuned)
| Metric | Base mean | Fine-tuned mean | Difference | t-test p | Significant (p<0.05) |
|---|---|---|---|---|---|
| bleu1 | 0.058 | 0.304 | +0.247 | 3.576e-179 | yes |
| bleu2 | 0.019 | 0.237 | +0.218 | 3.477e-144 | yes |
| bleu3 | 0.003 | 0.187 | +0.184 | 7.339e-109 | yes |
| bleu4 | 0.001 | 0.146 | +0.146 | 1.048e-75 | yes |
| rouge_l | 0.093 | 0.428 | +0.335 | 2.348e-267 | yes |
| meteor | 0.149 | 0.431 | +0.282 | 4.734e-183 | yes |
| cider | 0.000 | 0.956 | +0.956 | 1.625e-57 | yes |
| bertscore_f1 | 0.843 | 0.907 | +0.064 | 0 | yes |

See `metrics.json`, `per_disease.csv`, `hallucinations.csv`, and `figures/` for detail.
