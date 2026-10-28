"""
Stage 8 — Inference, and Stage 8b — Baseline capture.

WHY BASELINE CAPTURE IS A FIRST-CLASS STAGE, NOT A FLAG
    Fine-tuning without a baseline is an assertion. The same six held-out leaves,
    described by the untuned model and by the tuned model, is the control
    condition — it is what turns "the model learned to describe leaves" into
    evidence. It costs five minutes, it needs no training, and it is the figure
    that will carry a paper and a presentation. So `predict()` takes an adapter
    path of None and that is a supported, named mode, not an afterthought.

WHY SAMPLING RATHER THAN BEAM SEARCH
    The model was taught that five different descriptions are all correct for one
    image. Beam search returns the single most probable sentence and hides
    exactly the capability that was trained. Sampling shows it. Greedy decoding
    stays available for when a report needs one reproducible string.

INPUT   a split (val/test), a Run, optionally an adapter path
OUTPUT  runs/<id>/predictions/<split>_<base|finetuned>.json
            [{image_id, disease, image, generated[], references[5]}, ...]

CONNECTS TO
    Evaluation reads that file and never re-runs the model, so a metric can never
    change because a chart was redrawn.
"""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image

from .config import Config
from .io import write_json
from .models.base import VLMAdapter


def group_by_image(rows: List[Dict]) -> List[Dict]:
    """Fold the five caption rows of an image back into one record."""
    grouped: Dict[str, Dict] = {}
    for row in rows:
        entry = grouped.setdefault(row["image_id"], {
            "image_id": row["image_id"],
            "disease": row["disease"],
            "image": row["image"],
            "instruction": row["instruction"],
            "references": [],
        })
        entry["references"].append(row["caption"])
    return sorted(grouped.values(), key=lambda e: e["image_id"])


class InferenceEngine:
    """
    Loads a model once and generates for many images.

    Model-agnostic: input construction and decoding are delegated to the adapter,
    so this class needs no change when the VLM changes.
    """

    def __init__(self, cfg: Config, adapter: VLMAdapter,
                 adapter_path: Optional[Path] = None) -> None:
        self.cfg = cfg
        self.adapter = adapter
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.is_finetuned = adapter_path is not None

        source = str(adapter_path) if (adapter_path and Path(adapter_path).exists()) else None
        self.processor = adapter.load_processor(source)
        model = adapter.load_model(for_training=False)

        if adapter_path is not None:
            from peft import PeftModel
            if not Path(adapter_path).exists():
                raise FileNotFoundError(
                    f"adapter not found: {adapter_path} — train first")
            model = PeftModel.from_pretrained(model, str(adapter_path))
            model.eval()
        self.model = model

    @torch.no_grad()
    def generate(self, image: Image.Image, instruction: str,
                 greedy: bool = False) -> str:
        inputs = self.adapter.build_generate_inputs(
            image, instruction, self.processor, self.device)
        prompt_length = inputs["input_ids"].shape[1]

        g = self.cfg.generate
        kwargs: Dict = {"max_new_tokens": g.max_new_tokens,
                        "repetition_penalty": g.repetition_penalty}
        if greedy or not g.do_sample:
            kwargs.update(do_sample=False, num_beams=1)
        else:
            kwargs.update(do_sample=True, temperature=g.temperature, top_p=g.top_p)

        out = self.model.generate(**inputs, **kwargs)
        return self.adapter.decode_new_tokens(self.processor, out, prompt_length)

    def run(self, entries: List[Dict], greedy: bool = False,
            num_samples: Optional[int] = None, verbose: bool = True) -> List[Dict]:
        n = 1 if greedy else (num_samples or self.cfg.generate.num_samples)
        results: List[Dict] = []
        started = time.time()

        for i, entry in enumerate(entries, 1):
            image = Image.open(entry["image"]).convert("RGB")
            generated = [self.generate(image, entry["instruction"], greedy)
                         for _ in range(n)]
            results.append({**entry, "generated": generated})

            if verbose:
                print(f"\n[{i}/{len(entries)}] {entry['image_id']}  "
                      f"({entry['disease']})")
                for j, text in enumerate(generated, 1):
                    print(f"    gen{j}: {text}")
                print(f"    ref1: {entry['references'][0]}")

        if verbose:
            per_class = defaultdict(int)
            for r in results:
                per_class[r["disease"]] += 1
            print(f"\n  {len(results)} images in {(time.time() - started) / 60:.1f} min"
                  f" | {dict(sorted(per_class.items()))}")
        return results


def predict(cfg: Config, adapter: VLMAdapter, rows: List[Dict],
            output_file: Path, adapter_path: Optional[Path] = None,
            greedy: bool = False, num_samples: Optional[int] = None,
            limit: Optional[int] = None, verbose: bool = True) -> Dict:
    """Generate for a split and persist the result. Returns the payload."""
    entries = group_by_image(rows)
    if limit:
        entries = entries[:limit]

    engine = InferenceEngine(cfg, adapter, adapter_path)
    results = engine.run(entries, greedy=greedy, num_samples=num_samples,
                         verbose=verbose)

    payload = {
        "model": "finetuned" if adapter_path is not None else "base",
        "adapter": str(adapter_path) if adapter_path else None,
        "base_model": cfg.model.name,
        "instruction": cfg.data.instruction,
        "decoding": "greedy" if greedy else "sampling",
        "min_pixels": cfg.min_pixels,
        "max_pixels": cfg.max_pixels,
        "n_images": len(results),
        "results": results,
    }
    write_json(payload, output_file)
    return payload
