# patch_florence2.py
# Run this ONCE before training to fix the tokenizers compatibility bug.
# Place in: C:\Users\Admin\Desktop\VLM\florence\
#
# Run:
#   python patch_florence2.py

import os
import glob

def find_processing_file():
    cache_base = os.path.expanduser(
        r"~\.cache\huggingface\modules\transformers_modules\microsoft"
    )
    pattern = os.path.join(cache_base, "**", "processing_florence2.py")
    files   = glob.glob(pattern, recursive=True)
    if not files:
        pattern2 = os.path.join(cache_base, "**", "Florence*", "**", "processing_florence2.py")
        files    = glob.glob(pattern2, recursive=True)
    return files

def patch_file(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    if "PATCHED_FOR_TOKENIZERS_022" in content:
        print(f"Already patched: {path}")
        return True

    # The buggy line accesses tokenizer.additional_special_tokens via backend
    fixes = [
        (
            "tokenizer.additional_special_tokens +",
            "getattr(tokenizer, \'additional_special_tokens\', []) +"
        ),
        (
            "tokenizer.additional_special_tokens+",
            "getattr(tokenizer, \'additional_special_tokens\', [])+"
        ),
        (
            "self.tokenizer.additional_special_tokens +",
            "getattr(self.tokenizer, \'additional_special_tokens\', []) +"
        ),
    ]

    patched = False
    for old, new in fixes:
        if old in content:
            content = content.replace(old, new, 1)
            patched = True
            break

    if not patched:
        # Broader search: find any line with additional_special_tokens
        lines = content.split("\n")
        new_lines = []
        for line in lines:
            if "additional_special_tokens" in line and "getattr" not in line:
                # Replace .additional_special_tokens access with safe getattr
                import re
                fixed = re.sub(
                    r"(\w+)\.additional_special_tokens",
                    r"getattr(\1, \'additional_special_tokens\', [])",
                    line
                )
                new_lines.append(fixed)
                patched = True
                print(f"  Fixed line: {line.strip()}")
                print(f"  Became   : {fixed.strip()}")
            else:
                new_lines.append(line)
        content = "\n".join(new_lines)

    # Add patch marker
    content = "# PATCHED_FOR_TOKENIZERS_022\n" + content

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    return patched

def main():
    print("Searching for Florence-2 processor file...")
    files = find_processing_file()

    if not files:
        print("ERROR: processing_florence2.py not found in HuggingFace cache.")
        print("Try loading Florence-2 once first:")
        print("  python -c \"from transformers import AutoProcessor; AutoProcessor.from_pretrained(\'microsoft/Florence-2-base-ft\', trust_remote_code=True)\"")
        return

    print(f"Found {len(files)} file(s):")
    for f in files:
        print(f"  {f}")
        result = patch_file(f)
        if result:
            print(f"  Patched successfully.")
        else:
            print(f"  Could not find the buggy line - may already be fixed.")

    print("\nDone. Now run: python quick_train_florence2.py")

if __name__ == "__main__":
    main()
