"""Validation-split answers of the fine-tuned tomato model for several training snapshots.

Uses the evaluation's own generation settings (plantdx.evaluation.inference_runner: the same chat
template, greedy decoding, 128 new tokens, same prompt) on the validation split, sorted the same way.
The one difference from the evaluation's ``_generate`` is that the token-confidence value is not
computed: in mlx-vlm 0.6.4 ``logprobs`` is the last token's full-vocabulary array, and converting it
element by element took about 3.5 s per image (outside the timed part) without changing the answer.
Only the fine-tuned model is run: the base model does not depend on the snapshot. One JSONL per
snapshot; finished snapshots are skipped, so the script can be restarted.

Usage (MLX env, from the project folder, with the 200-step snapshots in checkpoints/qwen25vl_tomato_qlora/;
    writes selection/val_predictions/ and selection/status.txt):
    python selection/select_generate.py 16201 16000 14000 ...
    (16201 = the final adapters.safetensors; other numbers = NNNNNNN_adapters.safetensors snapshots)
"""

import json
import os
import shutil
import sys
import time
from pathlib import Path

from plantdx.evaluation.inference_runner import _MAX_NEW_TOKENS, _TEMPERATURE
from plantdx.training import seeds
from plantdx.training.data.label_map import load_label_map
from plantdx.training.inference import load_model
from plantdx.utils.io import read_jsonl

RUN = "qwen25vl_tomato_qlora"
CKPT = Path("checkpoints") / RUN
DATASET = Path("artifacts/training") / RUN / "dataset"
MODEL = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
OUT = Path("selection")
FINAL_ITER = 16201


def adapter_dir(iteration: int) -> Path:
    """A folder with adapter_config.json + adapters.safetensors (symlink to the snapshot)."""
    d = OUT / "adapters" / f"iter_{iteration:05d}"
    d.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(CKPT / "adapter_config.json", d / "adapter_config.json")
    src = CKPT / ("adapters.safetensors" if iteration == FINAL_ITER else f"{iteration:07d}_adapters.safetensors")
    if not src.is_file():
        raise SystemExit(f"missing snapshot {src}")
    link = d / "adapters.safetensors"
    if link.is_symlink() or link.exists():
        link.unlink()
    os.symlink(src.resolve(), link)
    return d


def generate_answer(loaded, image_path: str, instruction: str) -> tuple[str, float]:
    """inference_runner._generate without the token-confidence step: (text, elapsed_ms)."""
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    prompt = apply_chat_template(loaded.processor, loaded.config, instruction, num_images=1)
    start = time.perf_counter()
    result = generate(
        loaded.model,
        loaded.processor,
        prompt,
        image=image_path,
        max_tokens=_MAX_NEW_TOKENS,
        temperature=_TEMPERATURE,
        verbose=False,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return str(getattr(result, "text", result)).strip(), elapsed_ms


def main() -> None:
    iterations = [int(a) for a in sys.argv[1:]]
    seeds.apply(20260711)
    label_map = load_label_map("tomato")
    rows = sorted(read_jsonl(DATASET / "validation.jsonl"), key=lambda r: str(r["image"]))
    (OUT / "val_predictions").mkdir(parents=True, exist_ok=True)
    status = OUT / "status.txt"
    for it in iterations:
        out = OUT / "val_predictions" / f"iter_{it:05d}.jsonl"
        if out.exists():
            continue
        t0 = time.time()
        loaded = load_model(MODEL, adapter_path=str(adapter_dir(it)))
        records = []
        for row in rows:
            image_path = str(row["image"])
            class_name = Path(image_path).parent.name
            text, elapsed_ms = generate_answer(loaded, image_path, str(row["question"]))
            records.append({
                "image_id": f"{class_name}/{Path(image_path).name}",
                "disease_id": label_map.get(class_name, f"unknown:{class_name}"),
                "finetuned_prediction": text,
                "finetuned_runtime_ms": elapsed_ms,
            })
        tmp = out.with_suffix(".part")
        tmp.write_text("".join(json.dumps(r) + "\n" for r in records))
        tmp.rename(out)
        del loaded
        with status.open("a") as fh:
            fh.write(f"iter {it} done {len(records)} rows {time.time() - t0:.0f}s {time.strftime('%F %T')}\n")
    with status.open("a") as fh:
        fh.write(f"ALL DONE {time.strftime('%F %T')}\n")


if __name__ == "__main__":
    main()
