"""
augmentation.py

Image augmentation utilities for training data:
- Basic augmentations:
  - Flip (horizontal / vertical)
  - Rotate
  - Crop
  - Zoom
  - Shift
- Optional augmentations:
  - Brightness / contrast adjustment
  - Sharpening
  - Gaussian noise
  - CLAHE (contrast enhancement)
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Callable, Optional
import random


# --------- BASIC AUGMENTATIONS --------- #

def random_horizontal_flip(img: np.ndarray, p: float = 0.5) -> np.ndarray:
    if random.random() < p:
        return cv2.flip(img, 1)
    return img


def random_vertical_flip(img: np.ndarray, p: float = 0.1) -> np.ndarray:
    if random.random() < p:
        return cv2.flip(img, 0)
    return img


def random_rotate(
    img: np.ndarray, angle_range: Tuple[int, int] = (-15, 15), p: float = 0.7
) -> np.ndarray:
    if random.random() > p:
        return img

    h, w = img.shape[:2]
    angle = random.uniform(angle_range[0], angle_range[1])
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return rotated


def random_shift(
    img: np.ndarray,
    shift_ratio: Tuple[float, float] = (0.1, 0.1),
    p: float = 0.5,
) -> np.ndarray:
    if random.random() > p:
        return img

    h, w = img.shape[:2]
    max_dx = shift_ratio[0] * w
    max_dy = shift_ratio[1] * h

    dx = random.uniform(-max_dx, max_dx)
    dy = random.uniform(-max_dy, max_dy)

    M = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(
        img,
        M,
        (w, h),
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return shifted


def random_zoom(
    img: np.ndarray,
    zoom_range: Tuple[float, float] = (0.9, 1.1),
    p: float = 0.5,
) -> np.ndarray:
    if random.random() > p:
        return img

    h, w = img.shape[:2]
    zoom_factor = random.uniform(zoom_range[0], zoom_range[1])

    # Zoom in: crop center and resize
    if zoom_factor > 1.0:
        new_h = int(h / zoom_factor)
        new_w = int(w / zoom_factor)
        y1 = (h - new_h) // 2
        x1 = (w - new_w) // 2
        cropped = img[y1 : y1 + new_h, x1 : x1 + new_w]
        zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        # Zoom out: resize smaller and pad
        new_h = int(h * zoom_factor)
        new_w = int(w * zoom_factor)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        top = (h - new_h) // 2
        bottom = h - new_h - top
        left = (w - new_w) // 2
        right = w - new_w - left
        zoomed = cv2.copyMakeBorder(
            resized,
            top,
            bottom,
            left,
            right,
            borderType=cv2.BORDER_REFLECT_101,
        )
    return zoomed


def random_crop(
    img: np.ndarray,
    crop_size: Optional[Tuple[int, int]] = None,
    scale_range: Tuple[float, float] = (0.8, 1.0),
    p: float = 0.5,
) -> np.ndarray:
    if random.random() > p:
        return img

    h, w = img.shape[:2]

    if crop_size is not None:
        ch, cw = crop_size
        if ch > h or cw > w:
            return img
    else:
        scale = random.uniform(scale_range[0], scale_range[1])
        ch = int(h * scale)
        cw = int(w * scale)

    if ch == h and cw == w:
        return img

    y1 = random.randint(0, h - ch)
    x1 = random.randint(0, w - cw)

    cropped = img[y1 : y1 + ch, x1 : x1 + cw]
    resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    return resized


# --------- OPTIONAL PART, EXTRA AUGMENTATIONS... --------- #

def random_brightness_contrast(
    img: np.ndarray,
    brightness_range: Tuple[int, int] = (-30, 30),
    contrast_range: Tuple[float, float] = (0.8, 1.2),
    p: float = 0.5,
) -> np.ndarray:
    if random.random() > p:
        return img

    brightness = random.randint(brightness_range[0], brightness_range[1])
    contrast = random.uniform(contrast_range[0], contrast_range[1])

    img_float = img.astype(np.float32)
    img_bc = img_float * contrast + brightness
    img_bc = np.clip(img_bc, 0, 255).astype(np.uint8)
    return img_bc


def sharpen_image(img: np.ndarray, p: float = 0.3) -> np.ndarray:
    if random.random() > p:
        return img

    kernel = np.array(
        [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
        dtype=np.float32,
    )
    sharpened = cv2.filter2D(img, -1, kernel)
    return sharpened


def add_gaussian_noise(
    img: np.ndarray,
    mean: float = 0.0,
    std: float = 10.0,
    p: float = 0.3,
) -> np.ndarray:
    if random.random() > p:
        return img

    noise = np.random.normal(mean, std, img.shape).astype(np.float32)
    img_float = img.astype(np.float32)
    noisy = img_float + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy


def apply_clahe(
    img: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
    p: float = 0.3,
) -> np.ndarray:
    if random.random() > p:
        return img

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return enhanced


# --------- PIPELINE + DATASET AUGMENTATION --------- #

def augment_image_pipeline(img: np.ndarray) -> np.ndarray:
    """
    Define a pipeline of augmentations to apply to one image.
    You can tweak the order and probabilities above.
    """
    img = random_horizontal_flip(img, p=0.5)
    img = random_vertical_flip(img, p=0.1)
    img = random_rotate(img, angle_range=(-15, 15), p=0.7)
    img = random_shift(img, shift_ratio=(0.1, 0.1), p=0.5)
    img = random_zoom(img, zoom_range=(0.9, 1.1), p=0.5)
    img = random_crop(img, scale_range=(0.8, 1.0), p=0.5)

    # Optional / extra techniques:
    img = random_brightness_contrast(img, p=0.5)
    img = sharpen_image(img, p=0.3)
    img = add_gaussian_noise(img, p=0.3)
    img = apply_clahe(img, p=0.3)

    return img


def load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


def save_image(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def augment_dataset(
    input_root: str = "data/train",
    output_root: str = "data/train_aug",
    augmentations_per_image: int = 2,
) -> None:
   
    input_root = Path(input_root)
    output_root = Path(output_root)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    image_paths = [p for p in input_root.rglob("*") if p.suffix.lower() in exts]

    print(f"[INFO] Found {len(image_paths)} images in {input_root}")

    for img_path in image_paths:
        try:
            img = load_image(img_path)
        except ValueError as e:
            print(f"[WARN] {e}")
            continue

        rel_path = img_path.relative_to(input_root)
        class_dir = rel_path.parent  # e.g. "0" or "cat"
        stem = img_path.stem  # filename without extension
        ext = img_path.suffix

        for i in range(augmentations_per_image):
            aug_img = augment_image_pipeline(img)
            out_path = output_root / class_dir / f"{stem}_aug{i}{ext}"
            save_image(out_path, aug_img)

    print(f"[INFO] Augmented images saved to {output_root}")
