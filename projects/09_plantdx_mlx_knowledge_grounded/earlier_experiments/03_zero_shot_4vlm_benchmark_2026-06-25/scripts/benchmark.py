"""
scripts/benchmark.py

Runs every model in config.MODELS over every image in images/ using the
fixed prompt in config.PROMPT, via mlx-vlm. For each (model, image) pair
it records the raw text output, latency, tokens/sec, and peak memory.

Design notes:
  - Models are loaded and run one at a time, sequentially, and freed
    before the next model loads — important on a 24GB unified-memory
    machine where a 12B-parameter 4-bit model can use a large fraction
    of available memory.
  - If a model fails to load (missing weights, incompatible mlx-vlm
    version, OOM, etc.) the error is recorded for every image under that
    model rather than crashing the whole run, so a single problematic
    model doesn't block results for the others.
  - Per-image raw text is also dumped to results/<model>/<image_stem>.txt
    for quick manual/qualitative review without opening any JSON.

Outputs:
  json_results/<model_key>_results.json
  json_results/all_raw_results.json
  results/<model_key>/<image_stem>.txt   (raw text dumps)
"""

import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from scripts import utils


def load_model_safely(model_key, model_path):
    """Returns (model, processor, model_config, error_str_or_None)."""
    try:
        from mlx_vlm import load
    except ImportError as e:
        return None, None, None, f"mlx_vlm is not installed: {e}"

    try:
        load_mlx_config = None
        try:
            from mlx_vlm.utils import load_config as load_mlx_config
        except Exception:
            load_mlx_config = None

        utils.log(f"[{model_key}] Loading {model_path} ...")
        model, processor = load(model_path)
        model_cfg = None
        if load_mlx_config is not None:
            try:
                model_cfg = load_mlx_config(model_path)
            except Exception:
                model_cfg = None
        return model, processor, model_cfg, None
    except Exception as e:
        return None, None, None, f"{type(e).__name__}: {e}"


def free_model(model, processor):
    del model, processor
    try:
        import mlx.core as mx
        if hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
            mx.metal.clear_cache()
        elif hasattr(mx, "clear_cache"):
            mx.clear_cache()
    except Exception:
        pass


def main():
    config.ensure_directories()
    images = utils.list_images(config.IMAGES_DIR)
    if not images:
        utils.log(f"No images found in {config.IMAGES_DIR}.")
        utils.log("Copy your crop disease images into that folder and re-run.")
        return

    utils.log(f"Found {len(images)} images.")
    utils.log(f"Models to benchmark: {list(config.MODELS.keys())}")

    all_results = []

    for model_key, model_path in config.MODELS.items():
        utils.log("=" * 70)
        utils.log(f"MODEL: {model_key}  ({model_path})")
        utils.log("=" * 70)

        model, processor, model_cfg, load_error = load_model_safely(model_key, model_path)
        model_results = []

        if load_error:
            utils.log(f"  !! Could not load {model_key}: {load_error}")
            utils.log("  Skipping this model for all images. See README troubleshooting.")
            for img_path in images:
                model_results.append({
                    "model": model_key,
                    "model_path": model_path,
                    "image": img_path.name,
                    "raw_output": "",
                    "latency_sec": None,
                    "tokens_per_sec": None,
                    "peak_memory_gb": None,
                    "error": load_error,
                    "timestamp": datetime.now().isoformat(),
                })
        else:
            for i, img_path in enumerate(images, 1):
                utils.log(f"  [{i}/{len(images)}] {img_path.name}")
                gen = utils.run_mlx_vlm_generate(
                    model, processor, model_cfg, config.PROMPT, img_path,
                    max_tokens=config.MAX_TOKENS, temperature=config.TEMPERATURE,
                )
                record = {
                    "model": model_key,
                    "model_path": model_path,
                    "image": img_path.name,
                    "raw_output": gen["text"],
                    "latency_sec": gen["latency_sec"],
                    "tokens_per_sec": gen["tokens_per_sec"],
                    "peak_memory_gb": gen["peak_memory_gb"],
                    "error": gen["error"],
                    "timestamp": datetime.now().isoformat(),
                }
                model_results.append(record)

                out_txt = config.RESULTS_DIR / model_key / f"{img_path.stem}.txt"
                out_txt.write_text(gen["text"] or f"[ERROR] {gen['error']}", encoding="utf-8")

                if gen["error"]:
                    utils.log(f"      !! generation error: {gen['error']}")

            free_model(model, processor)

        out_path = config.JSON_RESULTS_DIR / f"{model_key}_results.json"
        utils.save_json(model_results, out_path)
        utils.log(f"  Saved {len(model_results)} records -> {out_path}")
        all_results.extend(model_results)

    combined_path = config.JSON_RESULTS_DIR / "all_raw_results.json"
    utils.save_json(all_results, combined_path)
    utils.log(f"\nAll done. {len(all_results)} total records -> {combined_path}")


if __name__ == "__main__":
    main()
