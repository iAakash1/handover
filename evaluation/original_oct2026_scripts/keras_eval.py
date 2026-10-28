"""Evaluation reports for the Dec 2025 / Jan 2026 Keras models.

Transfer-learning models (MobileNetV2, EfficientNetB3, ConvNeXt-Tiny) were trained with
ImageDataGenerator(validation_split=0.2) on the 10 PlantVillage tomato folders; Keras takes the
first 20% of each class's sorted file list as the validation subset, so that exact held-out subset is
rebuilt here with the same generator settings.

The custom CNN (32x32, 16 output units: '.DS_Store' + 15 PlantVillage classes from the macOS
listing) is evaluated on the 15-class PlantVillage pool with its own preprocessing, and separately on
the iitb_fellowship tomato test split.
Usage: python keras_eval.py <out_dir>
"""
import json
import os
import sys
import time

import cv2
import keras
import matplotlib
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = sys.argv[1]
MODELS = r"C:\Users\Admin\Desktop\iitb_fellowship\KAMAAKSHA\model"
RAW = r"C:\Users\Admin\Desktop\iitb_fellowship\data\raw\PlantVillage"
TOMATO_TEST = r"C:\Users\Admin\Desktop\iitb_fellowship\data\test"
os.makedirs(OUT, exist_ok=True)
TOMATO = sorted(d for d in os.listdir(RAW) if d.startswith("Tomato"))
ALL15 = sorted(d for d in os.listdir(RAW) if os.path.isdir(os.path.join(RAW, d)))


def save(name, title, classes, ys, ps, extra):
    cr = classification_report(ys, ps, labels=list(range(len(classes))), target_names=classes,
                               output_dict=True, digits=4, zero_division=0)
    cm = confusion_matrix(ys, ps, labels=list(range(len(classes)))).tolist()
    rep = {"model": title, "evaluated_on": time.strftime("%Y-%m-%d"), **extra, "n": len(ys),
           "accuracy": round(cr["accuracy"], 4) if "accuracy" in cr else round(float(np.mean(np.array(ys) == np.array(ps))), 4),
           "macro_precision": round(cr["macro avg"]["precision"], 4), "macro_recall": round(cr["macro avg"]["recall"], 4),
           "macro_f1": round(cr["macro avg"]["f1-score"], 4), "weighted_f1": round(cr["weighted avg"]["f1-score"], 4),
           "per_class": {c: {k: round(v, 4) for k, v in cr[c].items()} for c in classes},
           "classes": classes, "confusion_matrix": cm}
    d = os.path.join(OUT, name)
    os.makedirs(d, exist_ok=True)
    json.dump(rep, open(os.path.join(d, "evaluation_report.json"), "w"), indent=2)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(np.array(cm), cmap="Blues")
    short = [c.replace("Tomato__", "").replace("Tomato_", "").replace("___", " ").replace("_", " ")[:22] for c in classes]
    ax.set_xticks(range(len(classes)), short, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(classes)), short, fontsize=8)
    mx = max(map(max, cm)) or 1
    for i, row in enumerate(cm):
        for j, v in enumerate(row):
            ax.text(j, i, v, ha="center", va="center", fontsize=6, color="white" if v > mx / 2 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(f"{title}\nacc {rep['accuracy']:.4f}, macro-F1 {rep['macro_f1']:.4f}")
    fig.tight_layout(); fig.savefig(os.path.join(d, "confusion_matrix.png"), dpi=140); plt.close(fig)
    print(name, rep["n"], rep["accuracy"], rep["macro_f1"], flush=True)


# ---- transfer-learning models: rebuild the Keras validation subset on a tomato-only view
TL = [
    ("mobilenetv2_keras", "best_model_mobilenet.keras", "MobileNetV2 (Keras, ImageNet, 2-stage fine-tune)",
     keras.applications.mobilenet_v2.preprocess_input),
    ("efficientnetb3_keras", "best_model_efficientNetB3.keras", "EfficientNetB3 (Keras, ImageNet, 2-stage fine-tune)",
     keras.applications.efficientnet.preprocess_input),
    ("convnext_tiny_keras", "best_model_convnext.keras", "ConvNeXt-Tiny (Keras, ImageNet, 2-stage fine-tune)",
     keras.applications.convnext.preprocess_input),
]
gen_cls = tf.keras.preprocessing.image.ImageDataGenerator
for name, f, title, pre in TL:
    it = gen_cls(preprocessing_function=pre, validation_split=0.2).flow_from_directory(
        RAW, classes=TOMATO, target_size=(224, 224), batch_size=32, class_mode="categorical", subset="validation",
        shuffle=False)
    m = keras.models.load_model(os.path.join(MODELS, f), compile=False)
    t0 = time.time()
    probs = m.predict(it, verbose=0)
    save(name, title, TOMATO, it.classes.tolist(), probs.argmax(1).tolist(),
         {"checkpoint": f, "eval_set": "Keras validation_split=0.2 subset (first 20% of sorted files per class) of the "
                                     "10 PlantVillage tomato folders - held out from training, used for checkpoint selection",
          "ms_per_image": round(1000 * (time.time() - t0) / len(it.classes), 2)})


# ---- custom CNN (Dec 2025)
def cnn_pre(img):
    h, w = img.shape[:2]; s = min(h, w)
    img = img[(h - s) // 2:(h + s) // 2, (w - s) // 2:(w + s) // 2]
    img = cv2.resize(img, (32, 32))
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.filter2D(img, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype("float32") / 255.0


cnn_classes = [".DS_Store"] + ALL15


def run_cnn(fname, title, root, classes, tag, note):
    xs, ys = [], []
    for ci, c in enumerate(classes):
        for fn in sorted(os.listdir(os.path.join(root, c))):
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                img = cv2.imread(os.path.join(root, c, fn))
                if img is not None:
                    xs.append(cnn_pre(img)); ys.append(ci)
    m = keras.models.load_model(os.path.join(MODELS, fname), compile=False)
    probs = m.predict(np.stack(xs), batch_size=256, verbose=0)
    pred_names = [cnn_classes[i] for i in probs.argmax(1)]
    ps = [classes.index(n) if n in classes else -1 for n in pred_names]
    ps = [p if p >= 0 else (y + 1) % len(classes) for p, y in zip(ps, ys)]  # out-of-set prediction counts as wrong
    save(tag, title, classes, ys, ps, {"checkpoint": fname, "eval_set": note})


for fname, label in (("best_model_cnn.keras", "best"), ("final_model.keras", "final")):
    run_cnn(fname, f"Custom CNN 32x32 ({label} checkpoint)", RAW, ALL15, f"custom_cnn_{label}_plantvillage15",
            "All 15 PlantVillage class folders (20,638 images). The CNN's own 80/20 split was drawn from the macOS file "
            "order, so this pool includes its training images; see the tomato test-split row for a second view.")
    run_cnn(fname, f"Custom CNN 32x32 ({label} checkpoint)", TOMATO_TEST, sorted(os.listdir(TOMATO_TEST)),
            f"custom_cnn_{label}_tomato_test", "iitb_fellowship 70/15/15 tomato test split (2,411 images), 10 classes")
