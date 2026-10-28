# January demo for the ConvNeXt-Tiny model: quality gate, then predict sample.jpg.
# If model/best_model_convnext.keras is missing it trains a new one first (keras env).

import os
import cv2
import tensorflow as tf

from preprocessing.preprocess_inference import preprocess_for_inference
from model.predict import predict_disease
from predict import TOMATO_CLASSES, preprocess_transfer
from preprocessing.quality_checks import quality_checks
from model.train_convnext import train_model

IMAGE_PATH = "sample.jpg"
DATASET_DIR = os.environ.get("TOMATO_DATASET_DIR", "C:/Users/Admin/Desktop/iitb_fellowship/data/raw/PlantVillage")
MODEL_PATH = "model/best_model_convnext.keras"

# the label order the model was trained with (sorted tomato folder names)
CLASS_NAMES = TOMATO_CLASSES


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
    img = preprocess_transfer(img, "convnext")

    model = tf.keras.models.load_model(MODEL_PATH)
    result = predict_disease(model, img, CLASS_NAMES)

    print("\nPrediction Result (ConvNeXt)")
    print(result)


if __name__ == "__main__":

    if not os.path.exists(MODEL_PATH):
        print("ConvNeXt model not found. Training started...")
        train_model(DATASET_DIR)

    run_pipeline(IMAGE_PATH)
