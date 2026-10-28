# August 2026: SigLIP / OpenCLIP prototype for the TIH tomato dataset

**In short:** a code-only prototype (written 27 August 2026) that classifies TIH tomato images by matching each
image against text prompts with contrastive vision-language models (SigLIP and OpenCLIP), as a lighter
alternative to generative VLMs.

## What is here
- `configs/`: settings for each backbone (`siglip.yaml`, `openclip.yaml`) and the class prompts (`prompts.yaml`).
- `src/data/`: dataset loading, validation, splitting and transforms.
- `src/models/`: a common model interface with SigLIP and OpenCLIP implementations.
- `src/training/`: optimiser, loss, early stopping and checkpointing.

## Status
The pipeline was written but not trained. The team chose Qwen2.5-VL with QLoRA for the TIH dataset instead
(see `2026-09_September/plantdx_and_tih_tomato`: 92.76% test accuracy).
