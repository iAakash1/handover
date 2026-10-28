import cv2
import numpy as np
import os

# ------------------ BASIC CHECKS ------------------ #

def check_format(image_path):
    allowed_ext = (".jpg", ".jpeg", ".png")
    return image_path.lower().endswith(allowed_ext)


def check_resolution(img, min_width=224, min_height=224):
    h, w = img.shape[:2]
    return w >= min_width and h >= min_height


# ------------------ QUALITY CHECKS ------------------ #

def check_blur(img, threshold=80.0):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return lap_var >= threshold, lap_var


def check_brightness(img, min_val=50, max_val=200):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_val = np.mean(gray)
    return min_val <= mean_val <= max_val, mean_val


def check_contrast(img, min_std=20, max_std=80):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    std = np.std(gray)
    return min_std <= std <= max_std, std


def check_noise(img, threshold=120):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    noise = np.std(gray - cv2.GaussianBlur(gray, (5, 5), 0))
    return noise <= threshold, noise



def check_blank_image(img, threshold=5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    std = np.std(gray)
    return std > threshold, std


# ------------------ MASTER FUNCTION ------------------ #

def preprocess_for_inference(image_path):
    results = {}

    if not check_format(image_path):
        return False, {"reason": "Invalid format"}

    img = cv2.imread(image_path)
    if img is None:
        return False, {"reason": "Image read failed"}

    results["resolution"] = check_resolution(img)

    blur_ok, blur_val = check_blur(img)
    bright_ok, bright_val = check_brightness(img)
    contrast_ok, contrast_val = check_contrast(img)
    noise_ok, noise_val = check_noise(img)
    blank_ok, blank_val = check_blank_image(img)

    results.update({
        "blur_ok": blur_ok,
        "blur_value": blur_val,
        "brightness_ok": bright_ok,
        "brightness": bright_val,
        "contrast_ok": contrast_ok,
        "contrast": contrast_val,
        "noise_ok": noise_ok,
        "noise": noise_val,
        "blank_ok": blank_ok
    })

    final_decision = all([
        results["resolution"],
        blur_ok,
        bright_ok,
        contrast_ok,
        noise_ok,
        blank_ok
    ])

    return final_decision, results
