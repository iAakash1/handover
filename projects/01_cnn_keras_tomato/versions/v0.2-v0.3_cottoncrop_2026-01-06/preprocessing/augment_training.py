import cv2
import numpy as np
import random

def augment_for_training(img):
    augmented = []

    augmented.append(cv2.flip(img, 1))
    augmented.append(cv2.flip(img, 0))

    angle = random.randint(-20, 20)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
    augmented.append(cv2.warpAffine(img, M, (w, h)))

    factor = random.uniform(0.7, 1.3)
    augmented.append(np.clip(img * factor, 0, 1))

    return augmented
