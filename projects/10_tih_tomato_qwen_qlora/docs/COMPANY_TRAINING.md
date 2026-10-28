# Company/TIH tomato training

Company tomato images → class mapping → frozen train/validation/test split → Qwen2.5-VL QLoRA → validation → final test → best adapter.

Only the company package and its app integration implement this path. The research pipeline is separate.

## Run

From this repository, use the prepared project Python environment:

```powershell
.\.venv\Scripts\python.exe -m plantdx.cli company train --config configs/train/qwen25vl_tih_tomato_strong.yaml
.\.venv\Scripts\python.exe -m plantdx.cli company train --config configs/train/qwen25vl_tih_tomato_strong.yaml --resume
```

The Streamlit Training page launches the same CLI and strong configuration. Its Train button requires passing preflight. Rendering the page, preparing data, or running smoke never launches the official run.

The local environment reuses CUDA packages from the existing VLM and iitb_fellowship environments via a .pth file; retain those environments. The default system Python is not the configured CUDA interpreter.

## Data and labels

The raw dataset is read-only: 7,095 images, 6,545 usable after exact-content deduplication and conflicting-label removal. The frozen splits contain 5,247 train / 649 validation / 649 test images. Every existing image hash and its split order was preserved. Prepared splits are reused and checked, never silently regenerated. Changed images, split policies, labels, paths or prompts block training.

The class profile contains only raw folder, class id and display label. Each image is paired with the eleven-category classification question and its correct category name. Company inference displays the predicted label.

Training uses inverse-square-root class-frequency sampling with replacement, one original-sized epoch, and deterministic seed-plus-epoch ordering. Validation and test retain their natural distributions.

## Training and selection

Qwen/Qwen2.5-VL-7B-Instruct uses 4-bit NF4 with double quantization and BF16 compute. LoRA rank 32 / alpha 64 / dropout 0.05 adapts language q/k/v/o projections; the vision tower stays frozen. Learning rate is 5e-5, micro-batch 1, accumulation 8, cosine schedule, 3% warmup, paged AdamW 8-bit, and gradient checkpointing.

Validation loss and full validation classification metrics are computed each epoch. Checkpoints include adapter, optimizer, scheduler, RNG and Trainer state. Loss-based early stopping has patience 1 and a maximum of three epochs. Trainer restores its best loss checkpoint, then the final adapter is restored from the saved checkpoint with highest validation macro F1 (accuracy and loss break ties). No training-loss-based selection occurs.

The final adapter is reloaded, validation metrics are reported, and the full test split is evaluated once. Test identity is bound to the chosen adapter; completed test predictions can be reused after an interruption. Reports include accuracy, macro precision/recall/F1, weighted F1, per-class precision/recall/F1/support, confusion matrix, invalid outputs and overfitting warnings.

## Readiness and time budget

```powershell
.\.venv\Scripts\python.exe -m plantdx.cli company smoke --config configs/train/qwen25vl_tih_tomato_strong.yaml --keep
.\.venv\Scripts\python.exe -m plantdx.cli company preflight --config configs/train/qwen25vl_tih_tomato_strong.yaml --deep
```

Smoke uses 22 training images and eleven validation images, with no official test predictions. It saves after one optimizer step and resumes to complete exactly two total steps, then reloads the adapter and checks validation collation and generation. Readiness evidence is bound to the implementation, environment, configuration and frozen splits.

The measured estimate includes maximum-epoch optimization, epoch validation loss and generation, final validation/test generation, 15% throughput allowance and five minutes of loading/checkpoint overhead. The launch gate requires this to fit within 3.5 hours, reserving a further 30 minutes. A wall-clock callback requests validation, save and stop after three hours of training-loop time. Runtime estimates cannot guarantee hardware availability or sustained throughput.

## Outputs

- artifacts/company/<run>/dataset: frozen JSONL splits and manifest.
- checkpoints/<run>: selected adapter, processor, class mapping, dataset manifest, run configuration/identity, best checkpoint metadata, epoch checkpoints.
- logs/<run>: structured loss/validation/memory/checkpoint progress; Streamlit also captures terminal output in train.log.
- reports/<run>: smoke result, training summary, epoch validation metrics, final validation/test metrics and predictions, confusion matrices, held-out examples and quality warnings.

Full training starts only when the user executes the command or clicks Train.
