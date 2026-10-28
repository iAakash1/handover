# 10 · EfficientNet in the Keras SoyNet run (January 2026)

The provided SoyNet Keras pipeline (`projects/03_soybean_soynet/run_experiments.py`, models `efficientnetb0` and
`efficientnetb3`) trained ResNet50 and MobileNetV2, but the two EfficientNet models did not build. Only empty 6 KB
`.keras` files were saved; we removed them from the handover because they can't be loaded.

The PyTorch run of the same pipeline trained both: EfficientNetB3 **93.93%**, EfficientNetB0 **93.39%**
(`projects/03_soybean_soynet/results/experiments/`, usable with `predict_soynet.py`).
