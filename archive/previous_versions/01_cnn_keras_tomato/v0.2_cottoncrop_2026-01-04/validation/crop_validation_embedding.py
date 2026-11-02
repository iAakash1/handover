import cv2
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity

model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")

def extract_embedding(img):
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocess_input(img.astype("float32"))
    img = np.expand_dims(img, axis=0)
    return model.predict(img, verbose=0)

def crop_validation_embedding(img, reference_embeddings, threshold=0.7):
    emb = extract_embedding(img)
    sim = cosine_similarity(emb, reference_embeddings)
    if sim.max() < threshold:
        return False, "Embedding mismatch"
    return True, "Embedding validation passed"
