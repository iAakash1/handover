import cv2
import numpy as np

def gaussian_blur(img, k=5):
    return cv2.GaussianBlur(img, (k, k), 0)

def sharpen(img):
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    return cv2.filter2D(img, -1, kernel)
