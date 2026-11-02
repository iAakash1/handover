# Earlier versions of this project (kept exactly as they were)

| Version (report) | Date | Folder | What it is |
|---|---|---|---|
| v5.0 | 13 Apr 2026 | `v5.0_crop_app_2026-04-13` | First production pipeline (Crop_Disease_VLM `6b4e667`): CLIP ViT-B/32 OOD gate (0.27), parser, plant-mismatch validator, GradCAM, Streamlit app, plus the BLIP-2/Florence/Qwen scripts of that day |
| – | 9 Apr 2026 | `qwen_variant_2026-04-09` | An earlier variant of the Qwen2.5-VL prepare / fine-tune / evaluate / app scripts (the `other/` folder of Crop_Disease_VLM `5dcff36`) |

The maintained app is `../../../projects/06_qwen25vl_plantvillage/crop_disease_app` (the 13 Jul 2026 version, `6a8760b`, with a cross-platform config and
a setup checker); run it with `python run_app.py 2`.
