# main.py

import os
import cv2
import tensorflow as tf

from preprocessing.preprocess_inference import preprocess_for_inference
from model.predict import predict_disease
from model.train_transfer import train_model

# -----------------------------
# Configuration
# -----------------------------
IMAGE_PATH = "sample.jpg"
DATASET_DIR = "dataset/PlantVillage_Tomato"
MODEL_PATH = "model/best_model.keras"

# Automatically detect class names from dataset folders
CLASS_NAMES = sorted([
    d for d in os.listdir(DATASET_DIR)
    if os.path.isdir(os.path.join(DATASET_DIR, d))
])

# -----------------------------
# Inference Pipeline
# -----------------------------
def run_pipeline(image_path):
    """
    Complete inference pipeline:
    - Load image
    - Apply inference preprocessing
    - Load trained model
    - Predict disease class
    """

    img = cv2.imread(image_path)

    if img is None:
        print("Invalid image path")
        return

    try:
        img = preprocess_for_inference(img)
    except ValueError as e:
        print("Rejected:", e)
        return

    model = tf.keras.models.load_model(MODEL_PATH)
    result = predict_disease(model, img, CLASS_NAMES)

    print("\nPrediction Result")
    print(result)


# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":

    # Train model only if not already present
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Training started...")
        train_model(DATASET_DIR)

    # Run inference
    run_pipeline(IMAGE_PATH)
