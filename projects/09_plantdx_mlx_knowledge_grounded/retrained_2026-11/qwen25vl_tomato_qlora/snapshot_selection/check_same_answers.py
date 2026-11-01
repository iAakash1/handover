"""Check that select_generate.generate_answer gives the same answers as the evaluation's _generate.

Reruns the first N validation images with the final adapter and compares them with
selection/val_predictions/iter_16201.jsonl (written with inference_runner._generate).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from select_generate import DATASET, MODEL, adapter_dir, generate_answer  # noqa: E402

from plantdx.training import seeds
from plantdx.training.inference import load_model
from plantdx.utils.io import read_jsonl

n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
seeds.apply(20260711)
rows = sorted(read_jsonl(DATASET / "validation.jsonl"), key=lambda r: str(r["image"]))[:n]
old = [json.loads(line) for line in Path("selection/val_predictions/iter_16201.jsonl").read_text().splitlines()][:n]
loaded = load_model(MODEL, adapter_path=str(adapter_dir(16201)))
same = 0
for row, o in zip(rows, old):
    text, _ = generate_answer(loaded, str(row["image"]), str(row["question"]))
    if text == o["finetuned_prediction"]:
        same += 1
    else:
        print("DIFF", o["image_id"], repr(o["finetuned_prediction"][:80]), repr(text[:80]))
print(f"{same}/{n} identical")
