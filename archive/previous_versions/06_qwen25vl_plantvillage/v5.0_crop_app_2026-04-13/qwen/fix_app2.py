# fix_app2.py
with open("05_app.py", encoding="utf-8") as f:
    txt = f.read()

txt = txt.replace("Qwen2VLForConditionalGeneration", "Qwen2_5_VLForConditionalGeneration")
txt = txt.replace('attn_implementation="flash_attention_2",', "")
txt = txt.replace("attn_implementation='flash_attention_2',", "")
txt = txt.replace("Flash Attention 2 · ", "bfloat16 · ")
txt = txt.replace("bfloat16 + LoRA r32 + Flash Attention 2", "bfloat16 + LoRA r32")

with open("05_app.py", "w", encoding="utf-8") as f:
    f.write(txt)

print("Fixed 05_app.py")
print("Qwen2_5_VL present:", "Qwen2_5_VLForConditionalGeneration" in txt)
print("flash_attn removed:", "flash_attention_2" not in txt)
