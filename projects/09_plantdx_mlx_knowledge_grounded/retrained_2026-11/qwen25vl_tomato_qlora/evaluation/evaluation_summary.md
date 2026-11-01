# Evaluation Summary: Base vs Fine-tuned (Qwen2.5-VL, tomato)

- Samples evaluated: **910**
- Base accuracy: **0.073** | Fine-tuned accuracy: **0.909**
- Base macro-F1: **0.034** | Fine-tuned macro-F1: **0.877**

## Statistical significance (paired, base vs fine-tuned)
| Metric | Base mean | Fine-tuned mean | Difference | t-test p | Significant (p<0.05) |
|---|---|---|---|---|---|
| bleu1 | 0.058 | 0.287 | +0.230 | 4.039e-180 | yes |
| bleu2 | 0.019 | 0.220 | +0.201 | 2.054e-140 | yes |
| bleu3 | 0.003 | 0.173 | +0.170 | 1.31e-106 | yes |
| bleu4 | 0.001 | 0.135 | +0.135 | 9.144e-75 | yes |
| rouge_l | 0.093 | 0.405 | +0.312 | 8.118e-255 | yes |
| meteor | 0.149 | 0.420 | +0.271 | 5.332e-182 | yes |
| cider | 0.000 | 0.815 | +0.815 | 1.755e-53 | yes |
| bertscore_f1 | 0.843 | 0.903 | +0.059 | 1.26e-296 | yes |

See `metrics.json`, `per_disease.csv`, `hallucinations.csv`, and `figures/` for detail.
