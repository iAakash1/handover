"""Ask any of my fine-tuned vision-language models to diagnose one leaf photo.

I use this for quick demos: same model loading and generation settings as vlm_eval.py, just one image.
Run from this folder with the vlm environment (florence needs the florence environment):

    ..\\.venvs\\vlm\\Scripts\\python vlm_predict.py --model qwen_april --image path\\to\\leaf.jpg

Models (adapter folder in projects/):
    qwen_april   Qwen2.5-VL-7B + LoRA r32, April 2026 (98.50%)       06_qwen25vl_plantvillage/qwen
    plantdx      Qwen2.5-VL-7B + LoRA r32, PlantDx May 2026 (98.00%) 07_plantdx_plantvillage
    blip2_march  BLIP-2 OPT-2.7B + LoRA, March 2026 (89.00%)         04_llava_blip2/outputs
    blip2_june   BLIP-2 OPT-2.7B + LoRA, June re-run (89.00%)        04_llava_blip2/blip2_rerun_june
    llava        LLaVA-1.6-Mistral-7B + LoRA, March 2026 (2.00%)     04_llava_blip2/outputs
    plantvlm     Qwen2.5-VL-7B + LoRA r16 on 150 expert captions     08_plantvlm_expert_captions
    qwen_base    Qwen2.5-VL-7B without any adapter (8.50%)
    florence     Florence-2-base-ft without fine-tuning (0%)
The first run downloads the base model from Hugging Face (Qwen 16 GB, LLaVA 15 GB, BLIP-2 14 GB) unless it is
already in the Hugging Face cache.
"""
import argparse
import os

import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(HERE, "..", "projects")
INSTRUCTION = ("Analyze this plant leaf image and provide a detailed diagnosis. Include: plant name, disease or "
               "health condition, severity level, causal pathogen, and visible symptoms.")
MODELS = {
    "qwen_april": ("qwen", os.path.join(P, "06_qwen25vl_plantvillage", "qwen", "outputs", "qwen25vl_plantvillage", "best"), INSTRUCTION),
    "plantdx": ("qwen", os.path.join(P, "07_plantdx_plantvillage", "outputs", "checkpoints", "best"), INSTRUCTION),
    "plantvlm": ("qwen", os.path.join(P, "08_plantvlm_expert_captions", "runs", "run_002_fold0", "checkpoints", "best"),
                 "Describe the visible condition of this tomato leaf."),
    "qwen_base": ("qwen", None, INSTRUCTION),
    "blip2_march": ("blip2", os.path.join(P, "04_llava_blip2", "outputs", "blip2_plantvillage", "best"), None),
    "blip2_june": ("blip2", os.path.join(P, "04_llava_blip2", "blip2_rerun_june", "outputs", "blip2_plantvillage", "best"), None),
    "llava": ("llava", os.path.join(P, "04_llava_blip2", "outputs", "llava_plantvillage", "best"),
              "As a plant disease expert, analyze and describe this leaf image."),
    "florence": ("florence", None, None),
}


def load(kind, adapter):
    if kind == "qwen":
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        base_id = "Qwen/Qwen2.5-VL-7B-Instruct"
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(base_id, torch_dtype=torch.bfloat16, device_map="cuda")
    elif kind == "llava":
        from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor as AutoProcessor
        base_id = "llava-hf/llava-v1.6-mistral-7b-hf"
        model = LlavaNextForConditionalGeneration.from_pretrained(base_id, torch_dtype=torch.bfloat16, device_map="cuda")
    elif kind == "blip2":
        from transformers import Blip2ForConditionalGeneration, Blip2Processor as AutoProcessor
        base_id = "Salesforce/blip2-opt-2.7b"
        model = Blip2ForConditionalGeneration.from_pretrained(base_id, torch_dtype=torch.bfloat16, device_map="cuda")
    else:
        from transformers import AutoModelForCausalLM, AutoProcessor
        base_id = "microsoft/Florence-2-base-ft"
        model = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.float32, trust_remote_code=True).to("cuda")
        return model.eval(), AutoProcessor.from_pretrained(base_id, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(adapter or base_id)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    return model.eval(), processor


def generate(kind, model, proc, image, instruction):
    with torch.no_grad():
        if kind in ("qwen", "llava"):
            content = [{"type": "image", "image": image} if kind == "qwen" else {"type": "image"},
                       {"type": "text", "text": instruction}]
            text = proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                            add_generation_prompt=True)
            inputs = proc(text=[text], images=[image], return_tensors="pt").to("cuda")
            out = model.generate(**inputs, max_new_tokens=160, num_beams=4, repetition_penalty=2.5,
                                 no_repeat_ngram_size=6, early_stopping=True)
            return proc.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        if kind == "blip2":
            inputs = proc(images=image, return_tensors="pt").to("cuda", torch.bfloat16)
            out = model.generate(**inputs, max_new_tokens=80, num_beams=4, repetition_penalty=3.0,
                                 no_repeat_ngram_size=8, length_penalty=0.6, early_stopping=True)
            return proc.decode(out[0], skip_special_tokens=True).strip()
        inputs = proc(text="<MORE_DETAILED_CAPTION>", images=image, return_tensors="pt").to("cuda")
        out = model.generate(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
                             max_new_tokens=160, num_beams=3)
        return proc.batch_decode(out, skip_special_tokens=True)[0].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, default="qwen_april")
    ap.add_argument("--image", required=True)
    args = ap.parse_args()
    kind, adapter, instruction = MODELS[args.model]
    if adapter and not os.path.isdir(adapter):
        raise SystemExit(f"adapter not found: {adapter}\nrun: python setup/download_models.py")
    model, proc = load(kind, adapter)
    print(generate(kind, model, proc, Image.open(args.image).convert("RGB"), instruction))


if __name__ == "__main__":
    main()
