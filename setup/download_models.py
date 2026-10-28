"""Download every trained model into the folder its project code expects.

The weights are too big for a normal git repo, so I keep them in my TIH-IoT repository (Git LFS) and pull
them from there. Each file is checked against the SHA-256 in models_manifest.json, and files that are
already present and correct are skipped, so it is safe to run this again at any time.

Usage (from the repository root, any Python 3.9+, no extra packages needed):
    python setup/download_models.py                 # the final model of every project (~3.6 GB)
    python setup/download_models.py 06_qwen25vl      # only paths containing this text
    python setup/download_models.py --all           # also every older version and per-epoch checkpoint (+~2.6 GB)
"""
import hashlib
import json
import os
import shutil
import sys
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "setup", "models_manifest.json")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    manifest = json.load(open(MANIFEST))
    base = f"https://media.githubusercontent.com/media/{manifest['repository']}/{manifest['branch']}/"
    args = sys.argv[1:]
    everything = "--all" in args
    args = [a for a in args if a != "--all"]
    only = args[0] if args else ""
    # "optional" files are older versions and per-epoch checkpoints; the apps only need the final models
    files = [f for f in manifest["files"] if only in f["path"] and (everything or not f.get("optional"))]
    print(f"{len(files)} model file(s) to check")
    failed = 0
    ready = {}  # sha256 -> a local copy already checked; older versions reuse the same weights
    for i, f in enumerate(files, 1):
        dst = os.path.join(ROOT, f["path"])
        if os.path.exists(dst) and os.path.getsize(dst) == f["bytes"] and sha256(dst) == f["sha256"]:
            print(f"[{i}/{len(files)}] ok      {f['path']}")
            ready[f["sha256"]] = dst
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if f["sha256"] in ready:
            shutil.copyfile(ready[f["sha256"]], dst)
            print(f"[{i}/{len(files)}] copied  {f['path']} (same file as {os.path.relpath(ready[f['sha256']], ROOT)})")
            continue
        url = base + urllib.parse.quote(f["source"])
        print(f"[{i}/{len(files)}] fetch   {f['path']} ({f['bytes'] / 1e6:.0f} MB)", flush=True)
        tmp = dst + ".part"
        try:
            urllib.request.urlretrieve(url, tmp)
            if sha256(tmp) != f["sha256"]:
                raise ValueError("checksum mismatch")
            os.replace(tmp, dst)
            ready[f["sha256"]] = dst
        except Exception as e:  # keep going so one bad file does not stop the rest
            failed += 1
            print(f"           FAILED: {e}")
            if os.path.exists(tmp):
                os.remove(tmp)
    print("all models ready" if not failed else f"{failed} file(s) failed - run the script again")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
