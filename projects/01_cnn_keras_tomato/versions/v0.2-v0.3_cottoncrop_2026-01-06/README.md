# CottonCrop: preprocessing module and Keras transfer learning (Dec 2025 – Jan 2026)

**What is here:** Copy of [iAakash1/CottonCrop](https://github.com/iAakash1/CottonCrop) without its image folder. `preprocessing/` is the December Week-2 work (training preprocessing, augmentation, quality checks, inference gate); `validation/` rejects non-leaf images; `model/` has EfficientNetB3, MobileNetV2 and ConvNeXt-Tiny Keras models (trained on PlantVillage tomato, despite the repository name).

**Results:** Held-out validation: EfficientNetB3 93.59%, MobileNetV2 90.90%, ConvNeXt-Tiny 96.15%.

**Finding:** The image-quality gate stops blurry, dark or noisy photos before they reach the model.

---

*Original README:*

# CottonCrop