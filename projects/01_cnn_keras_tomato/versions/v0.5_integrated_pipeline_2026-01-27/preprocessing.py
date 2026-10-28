import cv2
import numpy as np
import random

def augment_for_training(img):
    aug = []

    aug.append(cv2.flip(img, 1))
    aug.append(cv2.flip(img, 0))

    angle = random.randint(-20, 20)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
    aug.append(cv2.warpAffine(img, M, (w, h)))

    factor = random.uniform(0.7, 1.3)
    aug.append(np.clip(img * factor, 0, 1))

    return aug

def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def to_hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

def gaussian_blur(img, k=5):
    return cv2.GaussianBlur(img, (k, k), 0)

def median_blur(img, k=5):
    return cv2.medianBlur(img, k)

def sharpen(img):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)

def sobel_edges(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return cv2.magnitude(sx, sy)

def blur_check(img, threshold=100):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() > threshold

def brightness_check(img, low=40, high=220):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean = gray.mean()
    return low <= mean <= high

def contrast_check(img, threshold=20):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray.std() > threshold

def noise_check(img, threshold=25):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    noise = cv2.Laplacian(gray, cv2.CV_64F).std()
    return noise < threshold

def resolution_check(img, min_size=128):
    h, w = img.shape[:2]
    return h >= min_size and w >= min_size

def quality_checks(img):
    if not blur_check(img):
        return False, "Image too blurry"
    if not brightness_check(img):
        return False, "Poor brightness"
    if not contrast_check(img):
        return False, "Low contrast"
    if not noise_check(img):
        return False, "Too noisy"
    if not resolution_check(img):
        return False, "Low resolution"
    return True, "Quality passed"


def preprocess_for_training(img, size=(32, 32)):
    h, w = img.shape[:2]
    side = min(h, w)

    # Center crop
    img = img[
        (h - side)//2:(h + side)//2,
        (w - side)//2:(w + side)//2
    ]

    img = cv2.resize(img, size)
    img = gaussian_blur(img)
    img = sharpen(img)
    img = bgr_to_rgb(img)

    img = img.astype("float32") / 255.0
    return img

def preprocess_for_inference(img):
    valid, msg = quality_checks(img)
    if not valid:
        raise ValueError(msg)

    return preprocess_for_training(img)
