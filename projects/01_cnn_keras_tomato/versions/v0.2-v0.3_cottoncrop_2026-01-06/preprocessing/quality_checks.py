import cv2
import numpy as np

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
