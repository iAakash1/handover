import cv2
import numpy as np

def preprocess_image(
    img,
    target_size=(224, 224),
    normalize=True,
    convert_to_gray=False
):
    # Resize
    img = cv2.resize(img, target_size)
    # Color conversion
    if convert_to_gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.expand_dims(img, axis=-1)
    # Normalize
    if normalize:
        img = img.astype("float32") / 255.0

    return img

def augment_image(img):
    # Horizontal flip
    if np.random.rand() < 0.5:
        img = cv2.flip(img, 1)

    # Brightness change
    factor = 0.8 + np.random.rand() * 0.4
    img = np.clip(img * factor, 0, 1)

    # Small rotation
    if np.random.rand() < 0.5:
        h, w = img.shape[:2]
        angle = np.random.uniform(-10, 10)
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
        img = cv2.warpAffine(img, M, (w, h))

    return img

def training_preprocess(img):
    img = preprocess_image(img)
    img = augment_image(img)
    return img

def inference_preprocess(img):
    img = preprocess_image(img)
    return img
