import os
import cv2
import numpy as np
from validation.crop_validation_embedding import extract_embedding

REFERENCE_IMAGE_DIR = "/Users/kharulkamaaksha/Desktop/PlantVillage_split/train"
OUTPUT_PATH = "reference/reference_crop_embeddings.npy"
MAX_IMAGES_PER_CLASS = 5


def generate_reference_embeddings():
    embeddings = []

    for cls in os.listdir(REFERENCE_IMAGE_DIR):
        cls_path = os.path.join(REFERENCE_IMAGE_DIR, cls)
        if not os.path.isdir(cls_path):
            continue

        images = os.listdir(cls_path)[:MAX_IMAGES_PER_CLASS]

        for img_name in images:
            img_path = os.path.join(cls_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            emb = extract_embedding(img)
            if emb is not None:
                embeddings.append(emb)

        print(f"{cls}: done")

    # Convert to array
    embeddings = np.array(embeddings, dtype="float32")

    if len(embeddings) == 0:
        raise RuntimeError("No embeddings generated")

    # Normalize ONCE (CRITICAL)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.clip(norms, 1e-8, None)

    # Save ONCE
    np.save(OUTPUT_PATH, embeddings)

    print("\nReference embeddings saved")
    print("Shape:", embeddings.shape)
    print("Mean norm:", np.mean(np.linalg.norm(embeddings, axis=1)))


if __name__ == "__main__":
    generate_reference_embeddings()
