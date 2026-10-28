"""Predict the tomato leaf disease in one image with any of my Keras models.

This is the entry point I use for demos. Each model gets the same preprocessing it saw during training:
  - the transfer-learning models (MobileNetV2, EfficientNetB3, ConvNeXt-Tiny) were trained with
    ImageDataGenerator + their own keras.applications preprocess_input at 224x224;
  - the small custom CNN was trained on 32x32 images after centre crop, Gaussian blur, sharpening and /255.

Run it from this folder with the keras environment:
    ..\\..\\.venvs\\keras\\Scripts\\python predict.py --image sample.jpg
    ..\\..\\.venvs\\keras\\Scripts\\python predict.py --image sample.jpg --model cnn
"""
import argparse
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# Sorted folder names of the 10 PlantVillage tomato classes - the label order every model was trained with.
TOMATO_CLASSES = [
    "Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_Late_blight", "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot", "Tomato_Spider_mites_Two_spotted_spider_mite", "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus", "Tomato__Tomato_mosaic_virus", "Tomato_healthy",
]
# The custom CNN was trained on all 15 PlantVillage folders listed on a Mac; '.DS_Store' got counted as class 0.
CNN_CLASSES = [".DS_Store", "Pepper__bell___Bacterial_spot", "Pepper__bell___healthy", "Potato___Early_blight",
               "Potato___Late_blight", "Potato___healthy"] + sorted(TOMATO_CLASSES)

MODELS = {
    "convnext": ("model/best_model_convnext.keras", "convnext"),
    "efficientnet": ("model/best_model_efficientNetB3.keras", "efficientnet"),
    "mobilenet": ("model/best_model_mobilenet.keras", "mobilenet_v2"),
    "cnn": ("model/best_model_cnn.keras", None),
}


def preprocess_cnn(img_bgr):
    # same steps as preprocessing/preprocess_training.py, just at 32x32
    h, w = img_bgr.shape[:2]
    s = min(h, w)
    img = img_bgr[(h - s) // 2:(h + s) // 2, (w - s) // 2:(w + s) // 2]
    img = cv2.resize(img, (32, 32))
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.filter2D(img, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype("float32") / 255.0


def preprocess_transfer(img_bgr, family):
    import keras
    img = cv2.cvtColor(cv2.resize(img_bgr, (224, 224), interpolation=cv2.INTER_NEAREST), cv2.COLOR_BGR2RGB)
    pre = getattr(keras.applications, family).preprocess_input
    return pre(img.astype("float32"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=os.path.join(HERE, "sample.jpg"))
    ap.add_argument("--model", choices=MODELS, default="convnext")
    ap.add_argument("--quality-check", action="store_true", help="reject blurry/dark/noisy photos first")
    args = ap.parse_args()

    import keras
    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"could not read image: {args.image}")
    if args.quality_check:
        from preprocessing.quality_checks import quality_checks
        ok, msg = quality_checks(img)
        print("quality check:", msg)
        if not ok:
            return

    path, family = MODELS[args.model]
    model = keras.models.load_model(os.path.join(HERE, path), compile=False)
    x = preprocess_cnn(img) if family is None else preprocess_transfer(img, family)
    probs = model.predict(x[None], verbose=0)[0]
    classes = CNN_CLASSES if family is None else TOMATO_CLASSES
    top = np.argsort(probs)[::-1][:3]
    print(f"model: {args.model}   image: {args.image}")
    for i in top:
        print(f"  {classes[i]:<45} {probs[i]:.3f}")


if __name__ == "__main__":
    main()
