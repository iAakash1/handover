# fix_clip.py
# Fixes CLIP loading in 03_inference.py and 04_evaluate.py
# Run: python fix_clip.py

for fname in ["03_inference.py", "04_evaluate.py", "05_app.py"]:
    try:
        with open(fname, encoding="utf-8") as f:
            txt = f.read()

        txt = txt.replace(
            "CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(\"cuda\")",
            "CLIPModel.from_pretrained(CLIP_MODEL_NAME, use_safetensors=True).to(\"cuda\")"
        )
        txt = txt.replace(
            "CLIPModel.from_pretrained(CLIP_MODEL_NAME).to('cuda')",
            "CLIPModel.from_pretrained(CLIP_MODEL_NAME, use_safetensors=True).to('cuda')"
        )

        with open(fname, "w", encoding="utf-8") as f:
            f.write(txt)

        found = "use_safetensors=True" in txt
        print(f"{'OK' if found else 'MISS'}  {fname}")

    except FileNotFoundError:
        print(f"SKIP {fname} (not found)")

print("\nDone. Run: python 03_inference.py")
