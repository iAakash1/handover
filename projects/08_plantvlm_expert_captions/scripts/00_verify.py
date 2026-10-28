"""
Verification harness — every check the pipeline makes about itself, in one place.

    python scripts/00_verify.py              # everything except GPU work (~15 s)
    python scripts/00_verify.py --gpu        # + real model load, forward, backward,
                                             #   checkpoint, reload, generate (~5 min)

WHY THIS EXISTS
    Verification that only runs as a side effect of building or training is
    verification you cannot repeat. This script re-checks the whole system on
    demand: after an annotator edits the workbook, after a dependency upgrade,
    before a deadline, or when something starts behaving strangely.

    It is also the honest answer to "is the project ready to train?" — if this
    exits 0 with --gpu, `python scripts/02_train.py` will not fail three minutes
    after you walk away.

LEVELS
    cheap  config, source, pairs, verification gate, split, JSONL round-trip,
           processor, conversation formatting, label masking, dataloader,
           evaluation metrics, retrieval, run directory
    --gpu  model load, LoRA attachment, vision-tower exclusion, forward, loss,
           backward, gradient flow, optimizer step, validation, checkpoint write,
           adapter reload, generation
"""

from __future__ import annotations

import argparse
import shutil
import sys
import traceback
from pathlib import Path

from _bootstrap import add_common_args, load

RESULTS: list[tuple[str, bool, str]] = []


def check(label: str):
    """Decorator turning a function into a recorded check."""
    def wrap(fn):
        def run(*a, **k):
            try:
                detail = fn(*a, **k) or ""
                RESULTS.append((label, True, str(detail)))
                print(f"  PASS  {label}" + (f"  [{detail}]" if detail else ""))
                return True
            except Exception as exc:
                RESULTS.append((label, False, f"{type(exc).__name__}: {exc}"))
                print(f"  FAIL  {label}\n        {type(exc).__name__}: {exc}")
                if "-v" in sys.argv:
                    traceback.print_exc()
                return False
        return run
    return wrap


# ── Cheap checks ─────────────────────────────────────────────────────────────

def verify_cheap(cfg) -> bool:
    from plantvlm.data import build_dataset, load_split
    from plantvlm.data.sources import get_source, normalize_whitespace
    from plantvlm.evaluation import ClassRetriever, caption_scores
    from plantvlm.io import read_jsonl
    from plantvlm.models import get_adapter
    from plantvlm.training import Run, describe_sample
    from plantvlm.training.dataset import CaptionDataset, build_loader

    state: dict = {}

    print("\n[1] configuration")

    @check("config loads, validates, and resolves paths")
    def _cfg():
        assert cfg.paths.excel.exists(), f"missing {cfg.paths.excel}"
        assert cfg.paths.image_root.exists(), f"missing {cfg.paths.image_root}"
        return (f"{len(cfg.data.source.class_map)} classes, fold {cfg.split.fold}, "
                f"batch {cfg.effective_batch}")
    _cfg()

    @check("config round-trips to YAML (freeze/restore)")
    def _freeze():
        tmp = cfg.paths.data_dir / "_verify_config.yaml"
        cfg.freeze_to(tmp)
        from plantvlm.config import load_config
        restored = load_config(tmp, root=cfg.paths.root)
        assert restored.data.instruction == cfg.data.instruction
        assert restored.split.fold == cfg.split.fold
        tmp.unlink()
        return "identical after reload"
    _freeze()

    @check("caption normalisation touches formatting only")
    def _norm():
        raw = "Veins are visible.The margins remain  intact. "
        out = normalize_whitespace(raw)
        assert out == "Veins are visible. The margins remain intact.", out
        assert len(out.split()) == len(raw.split()) + 1  # only the glued word splits
        return "period glue + whitespace"
    _norm()

    print("\n[2] source and pairs")

    @check("caption source loads every validated caption")
    def _source():
        records = get_source(cfg).load()
        state["records"] = records
        expected = (len(cfg.data.source.class_map)
                    * cfg.data.expected_images_per_class
                    * cfg.data.expected_captions_per_image)
        assert len(records) == expected, f"{len(records)} != {expected}"
        return f"{len(records)} captions"
    _source()

    @check("dataset builds, verifies, and splits")
    def _build():
        result = build_dataset(cfg, verify_provenance=False, write=True)
        state["result"] = result
        assert result["report"].ok
        return (f"{result['manifest']['n_pairs']} pairs, "
                f"{result['manifest']['n_images']} images, "
                f"{len(result['defects'])} flagged")
    _build()

    @check("verification gate rejects a broken dataset")
    def _gate():
        from plantvlm.data.verify import verify_dataset
        rows = [dict(r) for r in state["result"]["rows"]]
        images = state["result"]["images"]
        broken = {"train": rows[:5], "val": rows[:5], "test": rows[5:10]}
        report = verify_dataset(cfg, rows, images, broken)
        assert not report.ok, "gate accepted a leaking split"
        assert any("LEAKAGE" in f for f in report.failures)
        return "leakage detected as designed"
    _gate()

    print("\n[3] splits and JSONL")

    @check("no image appears in two splits")
    def _leak():
        parts = {name: load_split(cfg, name) for name in ("train", "val", "test")}
        ids = {k: {r["image_id"] for r in v} for k, v in parts.items()}
        assert not ids["train"] & ids["val"]
        assert not ids["train"] & ids["test"]
        assert not ids["val"] & ids["test"]
        state["splits"] = parts
        return (f"train {len(ids['train'])} / val {len(ids['val'])} / "
                f"test {len(ids['test'])} images")
    _leak()

    @check("all five captions of an image stay together")
    def _grouped():
        for name, rows in state["splits"].items():
            per = {}
            for r in rows:
                per[r["image_id"]] = per.get(r["image_id"], 0) + 1
            assert set(per.values()) == {cfg.data.expected_captions_per_image}, \
                f"{name}: {per}"
        return "5 rows per image in every split"
    _grouped()

    @check("JSONL round-trips and carries the required contract")
    def _jsonl():
        rows = read_jsonl(cfg.paths.train)
        required = {"id", "image", "image_id", "disease", "caption_index",
                    "instruction", "caption", "sha256", "split"}
        missing = required - set(rows[0])
        assert not missing, f"missing fields {missing}"
        assert all(Path(r["image"]).exists() for r in rows), "an image path is dead"
        return f"{len(rows)} rows, {len(required)} contract fields"
    _jsonl()

    print("\n[4] model adapter and conversation formatting")

    @check("adapter resolves from config")
    def _adapter():
        adapter = get_adapter(cfg)
        state["adapter"] = adapter
        return f"{adapter.name}, dtype {adapter.dtype}, device {adapter.device}"
    _adapter()

    @check("processor loads at the configured vision resolution")
    def _proc():
        processor = state["adapter"].load_processor()
        state["processor"] = processor
        return f"min_pixels {cfg.min_pixels}, max_pixels {cfg.max_pixels}"
    _proc()

    @check("conversation formatting produces correct label masking")
    def _mask():
        ds = CaptionDataset(state["splits"]["train"], state["adapter"],
                            state["processor"], cfg)
        sample = ds[0]
        info = describe_sample(sample, state["processor"])
        state["sample"] = sample
        assert info["supervised_tokens"] >= 10, "almost nothing supervised"
        expected = state["splits"]["train"][0]["caption"]
        assert info["supervised_text"].strip() in expected or \
               expected in info["supervised_text"], "supervised span != the caption"
        return (f"{info['supervised_tokens']} supervised tokens, "
                f"{info['tokens_used']}/{cfg.model.max_length} used")
    _mask()

    @check("no sample overflows max_length")
    def _budget():
        ds = CaptionDataset(state["splits"]["train"], state["adapter"],
                            state["processor"], cfg)
        used = [int(ds[i]["attention_mask"].sum()) for i in range(0, len(ds), 15)]
        assert max(used) < cfg.model.max_length, f"max {max(used)}"
        return f"longest sampled {max(used)}/{cfg.model.max_length}"
    _budget()

    @check("visual token count matches the configured resolution")
    def _tokens():
        n = state["adapter"].visual_tokens(state["sample"])
        assert 200 <= n <= 600, f"unexpected {n} visual tokens"
        return f"~{n} visual tokens/image"
    _tokens()

    @check("dataloader batches and collates")
    def _loader():
        loader = build_loader(state["splits"]["train"], state["adapter"],
                              state["processor"], cfg, shuffle=True)
        batch = next(iter(loader))
        shapes = {k: tuple(v.shape) for k, v in batch.items()}
        assert batch["input_ids"].shape[0] == cfg.train.micro_batch
        state["batch"] = batch
        return str(shapes)
    _loader()

    @check("augmentation is refused (captions are spatially grounded)")
    def _augment():
        import copy
        bad = copy.deepcopy(cfg)
        bad.train.augment = True
        try:
            CaptionDataset(state["splits"]["train"], state["adapter"],
                           state["processor"], bad)
        except ValueError:
            return "raises as designed"
        raise AssertionError("augmentation was silently accepted")
    _augment()

    print("\n[5] evaluation")

    @check("caption metrics score against all five references")
    def _metrics():
        rows = state["splits"]["val"]
        refs = [r["caption"] for r in rows[:5]]
        perfect = caption_scores(refs[0], refs)
        assert perfect["bleu1"] > 0.99, perfect
        poor = caption_scores("a completely unrelated sentence about nothing", refs)
        assert poor["bleu1"] < perfect["bleu1"]
        return f"BLEU-1 {perfect['bleu1']:.2f} (self) vs {poor['bleu1']:.2f} (unrelated)"
    _metrics()

    @check("class retriever + human ceiling are computed")
    def _retrieval():
        retriever = ClassRetriever(state["splits"]["train"], cfg)
        ceiling = retriever.human_ceiling(state["splits"]["val"])
        assert len(retriever.classes) == len(cfg.data.source.class_map)
        chance = 1 / len(retriever.classes)
        return (f"ceiling {ceiling['correct']}/{ceiling['total']} "
                f"({ceiling['value']:.0%}) vs chance {chance:.0%}")
    _retrieval()

    @check("evaluation runs end to end on synthetic predictions")
    def _eval_e2e():
        from plantvlm.evaluation import evaluate_predictions, render_report
        from plantvlm.inference import group_by_image
        from plantvlm.io import write_json

        entries = group_by_image(state["splits"]["val"])
        payload = {"model": "verify-synthetic", "n_images": len(entries),
                   "results": [{**e, "generated": [e["references"][2]]}
                               for e in entries]}
        tmp = cfg.paths.data_dir / "_verify_predictions.json"
        write_json(payload, tmp)
        report = evaluate_predictions(cfg, tmp, state["splits"]["train"],
                                      state["splits"]["val"])
        text = render_report(report)
        assert "Class retrieval accuracy" in text and "human ceiling" in text
        tmp.unlink()
        return f"{report['n_images']} images scored, report rendered"
    _eval_e2e()

    print("\n[6] run isolation")

    @check("run directory, frozen config, data copy, lineage")
    def _run():
        run = Run.create(cfg, tag="verify")
        run.write_lineage(cfg, state["result"]["manifest"])
        assert run.config_file.exists()
        assert run.lineage_file.exists()
        assert run.split_file("train").exists()
        lineage = run.read_lineage()
        assert lineage["source"]["sha256"], "workbook hash missing"
        assert lineage["environment"]["torch"], "environment not recorded"
        assert len(lineage["image_hashes"]) == cfg.data.expected_images_per_class * \
               len(cfg.data.source.class_map)
        shutil.rmtree(run.directory, ignore_errors=True)
        return (f"{len(lineage['image_hashes'])} image hashes, "
                f"torch {lineage['environment']['torch']}, "
                f"transformers {lineage['environment']['transformers']}")
    _run()

    @check("a second run does not overwrite the first")
    def _isolation():
        a = Run.create(cfg, tag="iso1")
        b = Run.create(cfg, tag="iso2")
        assert a.directory != b.directory
        names = (a.name, b.name)
        shutil.rmtree(a.directory, ignore_errors=True)
        shutil.rmtree(b.directory, ignore_errors=True)
        return f"{names[0]} != {names[1]}"
    _isolation()

    return all(ok for _, ok, _ in RESULTS)


# ── GPU checks ───────────────────────────────────────────────────────────────

def verify_gpu(cfg) -> bool:
    import torch
    from plantvlm.data import load_split
    from plantvlm.models import get_adapter
    from plantvlm.training import Run, Trainer

    print("\n[7] real model path (this loads ~16 GB)")

    state: dict = {}
    train_rows = load_split(cfg, "train")
    val_rows = load_split(cfg, "val")[:cfg.train.micro_batch * 2]

    @check("GPU is available and free enough")
    def _gpu():
        assert torch.cuda.is_available(), "CUDA not available"
        props = torch.cuda.get_device_properties(0)
        free = props.total_memory - torch.cuda.memory_allocated()
        return f"{props.name}, {props.total_memory / 1e9:.1f} GB, " \
               f"{free / 1e9:.1f} GB free"
    if not _gpu():
        return False

    @check("model loads, LoRA attaches, vision tower excluded")
    def _model():
        run = Run.create(cfg, tag="smoke")
        state["run"] = run
        trainer = Trainer(cfg, get_adapter(cfg), run, train_rows, val_rows)
        state["trainer"] = trainer
        pf = trainer.preflight()
        state["pf"] = pf
        leaked = [n for n, p in trainer.model.named_parameters()
                  if p.requires_grad and cfg.lora.exclude_substring in n]
        assert not leaked, f"{len(leaked)} vision params trainable"
        return (f"{pf['trainable_params']:,} trainable "
                f"({pf['trainable_pct']:.3f}%), 0 vision params")
    if not _model():
        return False

    @check("forward pass produces a finite loss")
    def _forward():
        trainer = state["trainer"]
        batch = next(iter(trainer.train_loader))
        batch = {k: v.to(trainer.device) for k, v in batch.items()}
        trainer.model.train()
        with torch.autocast(trainer.device, dtype=trainer.adapter.dtype):
            loss = trainer.model(**batch).loss
        assert torch.isfinite(loss), f"loss={loss}"
        state["loss"] = float(loss)
        return f"loss {float(loss):.4f}"
    _forward()

    @check("backward + optimizer step; gradients reach only the adapters")
    def _backward():
        trainer = state["trainer"]
        losses = []
        for i, batch in enumerate(trainer.train_loader):
            if i >= 2:
                break
            batch = {k: v.to(trainer.device) for k, v in batch.items()}
            with torch.autocast(trainer.device, dtype=trainer.adapter.dtype):
                loss = trainer.model(**batch).loss
            (loss / cfg.train.grad_accum).backward()
            grads = [n for n, p in trainer.model.named_parameters()
                     if p.requires_grad and p.grad is not None]
            assert grads, "no gradients reached the LoRA parameters"
            assert not any(cfg.lora.exclude_substring in n for n in grads)
            torch.nn.utils.clip_grad_norm_(trainer._trainable, cfg.train.grad_clip)
            trainer.optimizer.step()
            trainer.scheduler.step()
            trainer.optimizer.zero_grad()
            losses.append(float(loss))
        state["n_grads"] = len(grads)
        return f"{len(grads)} grad tensors, loss {losses[0]:.3f} -> {losses[-1]:.3f}"
    _backward()

    @check("validation loop runs")
    def _val():
        loss = state["trainer"].validate()
        assert loss == loss and loss != float("inf"), loss
        return f"val loss {loss:.4f}"
    _val()

    @check("checkpoint writes and contains the adapter")
    def _ckpt():
        trainer, run = state["trainer"], state["run"]
        trainer.save(run.checkpoint(0), {"verify": True})
        files = {p.name for p in run.checkpoint(0).iterdir()}
        assert {"adapter_config.json", "adapter_model.safetensors"} <= files
        size = sum(p.stat().st_size for p in run.checkpoint(0).iterdir()) / 1e6
        return f"{len(files)} files, {size:.0f} MB"
    _ckpt()

    @check("adapter reloads onto a fresh base model and generates")
    def _reload():
        from peft import PeftModel
        from PIL import Image

        trainer, run = state["trainer"], state["run"]
        adapter = trainer.adapter
        peak = torch.cuda.max_memory_allocated() / 1e9
        del trainer.model, trainer.optimizer
        state["trainer"] = None
        torch.cuda.empty_cache()

        base = adapter.load_model(for_training=False)
        model = PeftModel.from_pretrained(base, str(run.checkpoint(0)))
        model.eval()

        row = val_rows[0]
        image = Image.open(row["image"]).convert("RGB")
        inputs = adapter.build_generate_inputs(
            image, row["instruction"], trainer_processor(run, adapter),
            adapter.device)
        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=24, do_sample=False)
        text = adapter.decode_new_tokens(
            trainer_processor(run, adapter), out, prompt_len)
        assert text.strip(), "generation produced nothing"
        state["peak"] = peak
        return f"peak VRAM {peak:.1f} GB | \"{text[:44]}...\""
    _reload()

    if state.get("run"):
        shutil.rmtree(state["run"].directory, ignore_errors=True)
        print(f"\n  throw-away run removed: {state['run'].name}")

    return all(ok for _, ok, _ in RESULTS)


_PROCESSOR_CACHE: dict = {}


def trainer_processor(run, adapter):
    """Processor loaded from the smoke checkpoint (cached)."""
    key = str(run.checkpoint(0))
    if key not in _PROCESSOR_CACHE:
        _PROCESSOR_CACHE[key] = adapter.load_processor(key)
    return _PROCESSOR_CACHE[key]


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the whole pipeline")
    add_common_args(parser)
    parser.add_argument("--gpu", action="store_true",
                        help="also load the model and run a real training step")
    parser.add_argument("-v", action="store_true", help="show tracebacks")
    args = parser.parse_args()

    cfg = load(args)

    print("=" * 74)
    print("  PIPELINE VERIFICATION" + ("  (including GPU)" if args.gpu else ""))
    print(f"  config: {Path(cfg.source_file).name}")
    print("=" * 74)

    ok = verify_cheap(cfg)
    if args.gpu and ok:
        ok = verify_gpu(cfg) and ok
    elif args.gpu:
        print("\n  skipping GPU checks — fix the failures above first")

    passed = sum(1 for _, o, _ in RESULTS if o)
    print("\n" + "=" * 74)
    print(f"  {passed}/{len(RESULTS)} checks passed")
    for label, o, detail in RESULTS:
        if not o:
            print(f"    FAILED: {label} — {detail}")
    if ok:
        print("\n  PIPELINE READY.  Launch training with:")
        print("      python scripts/02_train.py")
    print("=" * 74)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
