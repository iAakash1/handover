# fix_app.py
# Run once after training: python fix_app.py
# Fixes 05_app.py: wrong VLM class + flash_attention_2 reference

with open("05_app.py") as f:
    txt = f.read()

# Fix 1: wrong class name in import
txt = txt.replace(
    "from transformers import (\n    Qwen2VLForConditionalGeneration,",
    "from transformers import (\n    Qwen2_5_VLForConditionalGeneration,"
)

# Fix 2: wrong class name in load_vlm()
txt = txt.replace(
    "base = Qwen2VLForConditionalGeneration.from_pretrained(",
    "base = Qwen2_5_VLForConditionalGeneration.from_pretrained("
)

# Fix 3: remove flash_attention_2 (not supported on this setup)
txt = txt.replace(
    "            torch_dtype=torch.bfloat16,\n            device_map=\"auto\",\n            attn_implementation=\"flash_attention_2\",",
    "            torch_dtype=torch.bfloat16,\n            device_map=\"auto\","
)

# Fix 4: update footer text to remove flash attn mention
txt = txt.replace(
    "\"<small>Qwen2.5-VL-7B-Instruct · LoRA r32 · Flash Attention 2 · \"",
    "\"<small>Qwen2.5-VL-7B-Instruct · LoRA r32 · bfloat16 · \""
)
txt = txt.replace(
    "\"**Strategy:** bfloat16 + LoRA r32 + Flash Attention 2 &nbsp;·&nbsp; \"",
    "\"**Strategy:** bfloat16 + LoRA r32 &nbsp;·&nbsp; \""
)

with open("05_app.py", "w") as f:
    f.write(txt)

# Verify
checks = [
    ("Qwen2_5_VLForConditionalGeneration", "Correct VLM class"),
    ("flash_attention_2" not in txt,       "flash_attn removed"),
]
print("Fix results:")
for result, label in checks:
    print(f"  {'OK' if result else 'MISS'}  {label}")

print("\nDone. Run: streamlit run 05_app.py")
