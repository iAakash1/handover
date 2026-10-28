"""Classify one tomato leaf photo with my fine-tuned ConvNeXt-Tiny (PyTorch, January 2026).

Same transforms as validation/test in notebooks/02_training_convnext.ipynb: resize to 224, ImageNet
normalisation. Run from this folder with the vlm environment:
    ..\\..\\.venvs\\vlm\\Scripts\\python predict.py --image path\\to\\leaf.jpg
"""
import argparse
import os

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

HERE = os.path.dirname(os.path.abspath(__file__))
CLASSES = [  # ImageFolder order = sorted folder names (also in splits/tomato_classes.txt order of the notebook)
    "Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_Late_blight", "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot", "Tomato_Spider_mites_Two_spotted_spider_mite", "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus", "Tomato__Tomato_mosaic_virus", "Tomato_healthy",
]
TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_model(device):
    model = models.convnext_tiny(weights=None)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, len(CLASSES))
    state = torch.load(os.path.join(HERE, "models", "convnext_tomato_best.pth"), map_location="cpu")
    model.load_state_dict(state)
    return model.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(device)
    x = TF(Image.open(args.image).convert("RGB"))[None].to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), 1)[0].cpu()
    for i in probs.argsort(descending=True)[:3]:
        print(f"  {CLASSES[i]:<45} {probs[i]:.3f}")


if __name__ == "__main__":
    main()
