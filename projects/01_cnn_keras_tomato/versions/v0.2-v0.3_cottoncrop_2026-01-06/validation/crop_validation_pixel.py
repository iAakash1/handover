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

    if green_ratio < 0.25:
        return False, "Low green content"
    if edge_ratio < 0.01:
        return False, "Low texture"

    return True, "Pixel validation passed"
