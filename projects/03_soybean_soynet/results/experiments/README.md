# SoyNet results per model (27 Jan 2026)

**What is here:** One folder per backbone with `metrics.json`, `confusion_matrix.json`, `history.json`, Grad-CAM images and the model (`model.pth` from the PyTorch run; `*.keras` from the Keras run). `final_report.md` summarises the Keras run.

**Results:** Keras ResNet50 95.16%; PyTorch EfficientNetB3 93.93%, EfficientNetB0 93.39%, MobileNetV2 92.50%, ResNet50 91.61%.

**Finding:** Precision/recall/F1 are for the Healthy (minority) class. The PyTorch models catch every healthy leaf (recall 1.0) but over-predict healthy.
