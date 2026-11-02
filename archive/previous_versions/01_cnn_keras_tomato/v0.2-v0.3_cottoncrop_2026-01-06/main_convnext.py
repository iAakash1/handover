import os
import cv2
import tensorflow as tf

from preprocessing.preprocess_inference import preprocess_for_inference
from model.predict import predict_disease
from model.train_convnext import train_model

IMAGE_PATH = "sample.jpg"
DATASET_DIR = "dataset/PlantVillage_Tomato"
MODEL_PATH = "model/best_model_convnext.keras"

CLASS_NAMES = sorted([
    d for d in os.listdir(DATASET_DIR)
    if os.path.isdir(os.path.join(DATASET_DIR, d))
])


def run_pipeline(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Invalid image path")
        return

    img = preprocess_for_inference(img)

    model = tf.keras.models.load_model(MODEL_PATH)
    result = predict_disease(model, img, CLASS_NAMES)

    print("\nPrediction Result (ConvNeXt)")
    print(result)


if __name__ == "__main__":

    if not os.path.exists(MODEL_PATH):
        print("ConvNeXt model not found. Training started...")
        train_model(DATASET_DIR)

    run_pipeline(IMAGE_PATH)
