# Handover guide: TIH tomato, 11 classes, Qwen2.5-VL QLoRA (project 10, ICAIN 2026 paper)

**1. What it does.** Classifies a tomato leaf or fruit photo into one of 11 TIH classes: aphid, early blight fruit,
fruit borer, healthy fruit, healthy leaf, late blight, leaf curl, leaf miner, mealy bugs, target leaf spot, white fly.
The model answers with the class name only. This is the work of the ICAIN 2026 paper.

**2. Architecture.** `Qwen/Qwen2.5-VL-7B-Instruct` loaded in 4-bit NF4 (bitsandbytes, double quantisation, bf16
compute) with a PEFT LoRA adapter on q/k/v/o of the language model; the vision tower is frozen. Code: the `plantdx`
package with `src/plantdx/company/` (audit, de-duplication, splits, training, `predict.py`, `evaluate.py`,
`external_evaluate.py`, adapter registry) and the Streamlit app (`streamlit_app.py`, `app/`).

**3. Dataset.** The private TIH `Tomato_12thAug26` set: 7,095 images → 6,545 after integrity checks, SHA-256
de-duplication and removal of images that appear under two classes; stratified 80/10/10 with seed 20260911:
5,247 train / 649 validation / 649 test (manifest: `checkpoints/tih_tomato_moreaccurate/dataset_manifest.json`).
External check: 176 labelled images from another source (`Dataset/testing`). **The images are not distributed;**
they are on the TIH workstation at `C:\Users\Admin\Desktop\TIH_IITB_August_2026\Tomato_12thAug26\Dataset`, and
access goes through the TIH-IoT mentors. The code looks for the dataset in the `candidate_roots` of
`configs/company_tomato_classes.yaml`; elsewhere pass `--dataset-root <folder with the 11 class folders>`.

**4. Training.** `tih_tomato_moreaccurate` (`configs/train/qwen25vl_tih_tomato_moreaccurate.yaml`): LoRA r64 /
alpha 128, LR 4e-5, up to 3 epochs (best checkpoint chosen on validation macro-F1: checkpoint-1908 at epoch 2.909),
532 px longest side, micro-batch 1 × gradient accumulation 8, paged AdamW 8-bit, class-balanced sampling,
answer-only loss; 193 min, 10.6 GB peak on the RTX 4500 Ada. The earlier `tih_tomato_qwen_qlora_strong` run used
r32 / alpha 64, LR 5e-5, 392 px.

**5. Weights.** `checkpoints/tih_tomato_moreaccurate/` (adapter files in git; `adapter_model.safetensors`, 161 MB) and
`checkpoints/tih_tomato_qwen_qlora_strong/`, fetched by `python setup\download_models.py 10_tih`; per-epoch
checkpoints (656, 1312, 1908) with `--all`. The app serves the adapter named in
`artifacts/runtime/active_adapter.json` (set to `tih_tomato_moreaccurate`).

**6. Install.** `.venvs\tih` (PyTorch 2.7.1 CUDA 11.8, transformers 5.3, PEFT 0.18, bitsandbytes 0.49, this package
installed editable); the models were trained with PyTorch 2.6.0 (CUDA 12.4), transformers 5.16.1, PEFT 0.20.0 and
bitsandbytes 0.50.2 (`checkpoints/tih_tomato_moreaccurate/run_identity.json`). **Needs an NVIDIA GPU** (4-bit NF4 via bitsandbytes); on a Mac the app opens but cannot diagnose.

**7. Inference.**
```powershell
python run_app.py 1                                                  # Streamlit app: Diagnose / Training / Adapter / Evaluation
cd projects\10_tih_tomato_qwen_qlora
$cfg = "configs/train/qwen25vl_tih_tomato_moreaccurate.yaml"; $py = "..\..\.venvs\tih\Scripts\python"
& $py -m plantdx.cli company predict --config $cfg --adapter checkpoints/tih_tomato_moreaccurate --image <photo>
```
How a prediction is made (`plantdx.company.predict.predict_image`): the photo is resized so its longer side is 532
px on the 28-px patch grid; the prompt asks for exactly one of the 11 categories; generation is greedy. The answer is
mapped to a class by `plantdx.company.evaluate.parse_prediction` (exact class name first, then the longest class
name found in the answer; otherwise "unparseable"). Confidence is the mean probability of the generated tokens; the
app marks results below 0.55 as low confidence.

**8. Reproduce evaluation.**
```powershell
& $py -m plantdx.cli company evaluate --config $cfg --adapter checkpoints/tih_tomato_moreaccurate --split test
& $py -m plantdx.cli company evaluate-external --config $cfg --adapter checkpoints/tih_tomato_moreaccurate `
      --dataset-root "<Dataset>/testing"
```
Reference: internal test **92.76%** accuracy, macro-F1 0.9378 (93.78%), 0.327 s per image
(`reports/tih_tomato_moreaccurate/test_finetuned/`); external 176 images **69.32%**, macro-F1 0.7195
(`reports/tih_tomato_moreaccurate_external/metrics.json`); "strong" run 87.98%
(`reports/tih_tomato_qwen_qlora_strong/test_finetuned/metrics.json`). The test split always runs in full (`--limit`
is refused on it); a new evaluation overwrites the run's reports unless `--run-name` is given.

**9. Modify / extend.** Each run is a YAML file in `configs/train/` (LoRA rank, image size, learning rate, epochs);
the class list and dataset location are in `configs/company_tomato_classes.yaml`. `company preflight` checks GPU,
dataset and split hashes before `company train`; `company activate --config <cfg> --adapter <dir>` switches the
adapter the app serves; `company status` lists the runs. A new crop: copy the class profile, list its class
folders, point a copy of the train config at it.

**10. Limitations.** Accuracy drops from 92.76% to 69.32% on photos from another source; late blight is the weakest
class there (F1 14.29%, recall 10%). CUDA only. The dataset is private.
