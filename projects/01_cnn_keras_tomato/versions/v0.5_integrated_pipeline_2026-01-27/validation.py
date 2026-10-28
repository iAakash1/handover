import cv2
import numpy as np

def crop_validation_pixel(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])

    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    green_ratio = (green_mask > 0).sum() / green_mask.size

    edges = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 100, 200)
    edge_ratio = (edges > 0).sum() / edges.size

    print("DEBUG → green_ratio:", round(green_ratio, 3))
    print("DEBUG → edge_ratio:", round(edge_ratio, 3))

    # Relaxed but safe thresholds (PlantVillage-tuned)
    if green_ratio < 0.08:
        return False, "Low green content"
    if edge_ratio < 0.008:
        return False, "Low texture"

    return True, "Pixel validation passed"

import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

# Load model once
model = MobileNetV2(
    input_shape=(224, 224, 3),
    weights="imagenet",
    include_top=False,
    pooling="avg"
)

def extract_embedding(img):
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocess_input(img.astype("float32"))
    img = np.expand_dims(img, axis=0)

    emb = model.predict(img, verbose=0)[0].astype("float32")

    norm = np.linalg.norm(emb)
    if norm == 0:
        return None

    return emb / norm


def crop_validation_embedding(img, reference_embeddings, threshold=0.45):
    emb = extract_embedding(img)
    if emb is None:
        return False, "Invalid embedding"

    sims = np.dot(reference_embeddings, emb)

    max_sim = float(np.max(sims))
    print("DEBUG → max embedding similarity:", round(max_sim, 3))

    if max_sim < threshold:
        return False, "Embedding mismatch"

    return True, "Embedding validation passed"

def validate_crop_image(img, reference_embeddings):
    ok, msg = crop_validation_pixel(img)
    if not ok:
        return False, msg

    ok, msg = crop_validation_embedding(img, reference_embeddings)
    if not ok:
        return False, msg

    return True, "Crop validated"