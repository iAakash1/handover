# patch_florence2.py  (updated - fixes processor AND config)
# Run ONCE before training.
# Place in: C:\Users\Admin\Desktop\VLM\florence\
# Run: python patch_florence2.py

import os, re, glob

def find_files(filename):
    cache_base = os.path.expanduser(r"~\.cache\huggingface\modules\transformers_modules\microsoft")
    return glob.glob(os.path.join(cache_base, "**", filename), recursive=True)

def patch_processing(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    if "PATCH1_DONE" in content:
        print("  [processing] Already patched")
        return
    original = content
    content = re.sub(
        r"(\w+)\.additional_special_tokens",
        r"getattr(\1, \'additional_special_tokens\', [])",
        content,
    )
    if content != original:
        content = "# PATCH1_DONE\n" + content
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print("  [processing] Patched OK:", path)
    else:
        print("  [processing] No changes needed")

def patch_config(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    if "PATCH2_DONE" in content:
        print("  [config] Already patched")
        return
    bad_line = "if self.forced_bos_token_id is None and kwargs.get"
    fix      = (
        "if not hasattr(self, \'forced_bos_token_id\'):\n"
        "            self.forced_bos_token_id = kwargs.pop(\'forced_bos_token_id\', None)\n"
        "        " + bad_line
    )
    if bad_line in content:
        content = content.replace(bad_line, fix, 1)
        content = "# PATCH2_DONE\n" + content
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print("  [config] Patched OK:", path)
    else:
        print("  [config] Target line not found - may not need patch")

def main():
    print("=" * 50)
    print("  Florence-2 Compatibility Patcher")
    print("=" * 50)

    print("\n[1/2] Patching processing_florence2.py ...")
    for f in find_files("processing_florence2.py"):
        patch_processing(f)

    print("\n[2/2] Patching configuration_florence2.py ...")
    for f in find_files("configuration_florence2.py"):
        patch_config(f)

    print("\nDone. Now run: python quick_train_florence2.py")

if __name__ == "__main__":
    main()