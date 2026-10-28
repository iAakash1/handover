"""Full evaluation report for iitb_fellowship/models/convnext_tomato_best.pth on its held-out test split.

Mirrors the transforms used in notebooks/02_training_convnext.ipynb.
Usage: python convnext_torch_eval.py <out_dir>
"""
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

ROOT = r"C:\Users\Admin\Desktop\iitb_fellowship"
OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
splits = {s: datasets.ImageFolder(os.path.join(ROOT, "data", s), transform=tf) for s in ("train", "val", "test")}
classes = splits["test"].classes

model = models.convnext_tiny(weights=None)
model.classifier[2] = nn.Linear(model.classifier[2].in_features, len(classes))
model.load_state_dict(torch.load(os.path.join(ROOT, "models", "convnext_tomato_best.pth"), map_location="cpu"))
model = model.cuda().eval()

report = {"model": "ConvNeXt-Tiny (torchvision, ImageNet-1k init, full fine-tune)",
          "checkpoint": "convnext_tomato_best.pth", "classes": classes,
          "split_sizes": {s: len(d) for s, d in splits.items()},
          "evaluated_on": time.strftime("%Y-%m-%d")}
for split in ("val", "test"):
    loader = DataLoader(splits[split], batch_size=64, shuffle=False, num_workers=0)
    ys, ps = [], []
    t0 = time.time()
    with torch.no_grad():
        for x, y in loader:
            ps.extend(model(x.cuda()).argmax(1).cpu().tolist())
            ys.extend(y.tolist())
    cr = classification_report(ys, ps, target_names=classes, output_dict=True, digits=4, zero_division=0)
    cm = confusion_matrix(ys, ps).tolist()
    report[split] = {
        "n": len(ys), "accuracy": round(cr["accuracy"], 4),
        "macro_precision": round(cr["macro avg"]["precision"], 4),
        "macro_recall": round(cr["macro avg"]["recall"], 4),
        "macro_f1": round(cr["macro avg"]["f1-score"], 4),
        "weighted_f1": round(cr["weighted avg"]["f1-score"], 4),
        "ms_per_image": round(1000 * (time.time() - t0) / len(ys), 2),
        "per_class": {c: {k: round(v, 4) for k, v in cr[c].items()} for c in classes},
        "confusion_matrix": cm,
    }
    if split == "test":
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(np.array(cm), cmap="Blues")
        short = [c.replace("Tomato_", "").replace("Tomato__", "").replace("_", " ")[:22] for c in classes]
        ax.set_xticks(range(len(classes)), short, rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(len(classes)), short, fontsize=8)
        for i, row in enumerate(cm):
            for j, v in enumerate(row):
                ax.text(j, i, v, ha="center", va="center", fontsize=7, color="white" if v > max(map(max, cm)) / 2 else "black")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"ConvNeXt-Tiny (PyTorch) test, acc {cr['accuracy']:.4f}")
        fig.tight_layout(); fig.savefig(os.path.join(OUT, "confusion_matrix_test.png"), dpi=150)

json.dump(report, open(os.path.join(OUT, "evaluation_report.json"), "w"), indent=2)
print(json.dumps({k: (v if k not in ("val", "test") else {kk: vv for kk, vv in v.items() if kk not in ("per_class", "confusion_matrix")}) for k, v in report.items()}, indent=2))
