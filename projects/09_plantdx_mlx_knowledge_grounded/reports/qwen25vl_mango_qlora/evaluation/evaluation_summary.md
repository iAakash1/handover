# Evaluation Summary: Base vs Fine-tuned (Qwen2.5-VL, mango)

- Samples evaluated: **200**
- Base accuracy: **0.175** | Fine-tuned accuracy: **0.820**
- Base macro-F1: **0.128** | Fine-tuned macro-F1: **0.811**

## Statistical significance (paired, base vs fine-tuned)
| Metric | Base mean | Fine-tuned mean | Difference | t-test p | Significant (p<0.05) |
|---|---|---|---|---|---|
| bleu1 | 0.060 | 0.300 | +0.240 | 1.285e-39 | yes |
| bleu2 | 0.023 | 0.223 | +0.200 | 1.779e-27 | yes |
| bleu3 | 0.004 | 0.177 | +0.173 | 4.59e-21 | yes |
| bleu4 | 0.001 | 0.144 | +0.144 | 5.957e-16 | yes |
| rouge_l | 0.098 | 0.423 | +0.325 | 2.206e-56 | yes |
| meteor | 0.160 | 0.454 | +0.294 | 1.081e-42 | yes |
| cider | 0.000 | 0.946 | +0.946 | 1.012e-10 | yes |
| bertscore_f1 | 0.843 | 0.907 | +0.064 | 2.181e-67 | yes |

See `metrics.json`, `per_disease.csv`, `hallucinations.csv`, and `figures/` for detail.
