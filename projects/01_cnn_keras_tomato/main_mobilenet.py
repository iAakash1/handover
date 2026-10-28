# main_mobilenet.py

import os
import cv2
import tensorflow as tf

from preprocessing.preprocess_inference import preprocess_for_inference
from model.predict import predict_disease
from predict import TOMATO_CLASSES, preprocess_transfer
from preprocessing.quality_checks import quality_checks
from model.train_mobilenet import train_model

# -----------------------------
# Configuration
# -----------------------------
IMAGE_PATH = "sample.jpg"
DATASET_DIR = os.environ.get("TOMATO_DATASET_DIR", "C:/Users/Admin/Desktop/iitb_fellowship/data/raw/PlantVillage")
MODEL_PATH = "model/best_model_mobilenet.keras"

# the label order the model was trained with (sorted tomato folder names)
CLASS_NAMES = TOMATO_CLASSES

# -----------------------------
# Pipeline
# -----------------------------
def run_pipeline(image_path):
    img = cv2.imread(image_path)

    if img is None:
        print("Invalid image path")
        return

    # same quality gate as before, then the preprocessing the model was trained with
    ok, msg = quality_checks(img)
    if not ok:
        print("Rejected:", msg)
        return
    img = preprocess_transfer(img, "mobilenet_v2")

    model = tf.keras.models.load_model(MODEL_PATH)
    result = predict_disease(model, img, CLASS_NAMES)

    print("\nPrediction Result (MobileNetV2)")
    print(result)


# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":

    if not os.path.exists(MODEL_PATH):
        print("MobileNetV2 model not found. Training started...")
        train_model(DATASET_DIR)

    run_pipeline(IMAGE_PATH)
