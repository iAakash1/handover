import cv2
from preprocessing.color_processing import bgr_to_rgb
from preprocessing.filters import gaussian_blur, sharpen

def preprocess_for_training(img, size=(224, 224)):
    h, w = img.shape[:2]
    side = min(h, w)

    img = img[
        (h - side) // 2 : (h + side) // 2,
        (w - side) // 2 : (w + side) // 2
    ]

    img = cv2.resize(img, size)
    img = gaussian_blur(img)
    img = sharpen(img)
    img = bgr_to_rgb(img)
    img = img.astype("float32") / 255.0

    return img
