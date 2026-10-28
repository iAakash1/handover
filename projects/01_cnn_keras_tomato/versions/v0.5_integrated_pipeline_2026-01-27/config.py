import os
import cv2
import numpy as np
import tensorflow as tf

from preprocessing import preprocess_for_inference
from validation.crop_validation import validate_crop_image
from model.predict import predict_disease

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

REFERENCE_EMB_PATH = os.path.join(
    BASE_DIR, "reference", "reference_crop_embeddings.npy"
)

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
CLASSES_TXT = os.path.join(DATASET_DIR, "classes.txt")

# cnn | mobilenet | efficientnet | convnext
SELECTED_MODEL = "cnn"  

MODEL_PATHS = {
    "cnn": "model/best_model.keras",
    "mobilenet": "model/best_model_mobilenet.keras",
    "efficientnet": "model/best_model_efficientNetB3.keras",
    "convnext": "model/best_model_convnext.keras"
}

MODEL_PATH = os.path.join(BASE_DIR, MODEL_PATHS[SELECTED_MODEL])

def load_class_names_from_model(model):
    if hasattr(model, "class_names"):
        return model.class_names
    return None

def load_class_names_from_file():
    if os.path.exists(CLASSES_TXT):
        with open(CLASSES_TXT, "r") as f:
            return [line.strip() for line in f if line.strip()]
    return None


def load_class_names_from_dataset():
    if os.path.exists(DATASET_DIR):
        return sorted([
            d for d in os.listdir(DATASET_DIR)
            if os.path.isdir(os.path.join(DATASET_DIR, d))
        ])
    return None


def get_class_names(model):
    for loader in (
        lambda: load_class_names_from_model(model),
        load_class_names_from_file,
        load_class_names_from_dataset
    ):
        classes = loader()
        if classes:
            return classes

    raise RuntimeError("Unable to resolve class names")

CONFIDENCE_THRESHOLD = 0.6

def run_pipeline(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Invalid image path")

    reference_embeddings = np.load(REFERENCE_EMB_PATH)

    valid, msg = validate_crop_image(img, reference_embeddings)
    if not valid:
        return {"status": "rejected", "reason": msg}

    processed_img = preprocess_for_inference(img)

    model = tf.keras.models.load_model(MODEL_PATH)

    class_names = get_class_names(model)

    result = predict_disease(
        model=model,
        processed_img=processed_img,
        class_names=class_names,
        confidence_threshold=CONFIDENCE_THRESHOLD
    )

    return result

if __name__ == "__main__":
    TEST_IMAGE = "/Users/kharulkamaaksha/Desktop/img1.jpg"
    print(run_pipeline(TEST_IMAGE))
