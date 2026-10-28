"""Healthy vs diseased soybean leaf, using the PyTorch models from the 27 January 2026 SoyNet run.

Run from this folder with the soynet environment:
    ..\\..\\.venvs\\soynet\\Scripts\\python predict_soynet.py --image path\\to\\leaf.jpg
    ..\\..\\.venvs\\soynet\\Scripts\\python predict_soynet.py --image leaf.jpg --model resnet50

Models: efficientnet_b3 (best, 93.93% test), efficientnet_b0, mobilenetv2, resnet50.
"""
import argparse
import os

import torch
from PIL import Image

from pytorch.dataset import get_transforms
from pytorch.models import ModelFactory

HERE = os.path.dirname(os.path.abspath(__file__))
CLASSES = ["Diseased", "Healthy"]  # ImageFolder order of the two training folders


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--model", default="efficientnet_b3", choices=list(ModelFactory.MODEL_MAP))
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # weights come from the checkpoint, so no ImageNet download is needed here
    model = ModelFactory.create_model(args.model, pretrained=False, freeze=False)
    ckpt = torch.load(os.path.join(HERE, "results", "experiments", args.model, "model.pth"), map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    x = get_transforms(train=False)(Image.open(args.image).convert("RGB"))[None].to(device)
    with torch.no_grad():
        out = model(x)
    # the classifier ends in one logit (binary, sigmoid); the positive class is index 1 = Healthy
    p_healthy = torch.sigmoid(out.reshape(-1)[0]).item()
    for name, p in zip(CLASSES, (1 - p_healthy, p_healthy)):
        print(f"  {name:<10} {p:.3f}")


if __name__ == "__main__":
    main()
