"""
quality_checks.py

Utilities to evaluate image quality in the dataset:
- Blur detection
- Brightness / contrast analysis
- Noise estimation
- Dataset-wide scanning + flagging low-quality images
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional


def load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


def compute_blur_score(img: np.ndarray) -> float:

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def compute_brightness_contrast(img: np.ndarray) -> Tuple[float, float]:

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    contrast = float(gray.std())
    return brightness, contrast


def estimate_noise_level(img: np.ndarray) -> float:

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    noise = gray.astype(np.float32) - blurred.astype(np.float32)
    noise_std = float(noise.std())
    return noise_std


def evaluate_image(
    path: Path,
    blur_threshold: float,
    brightness_range: Tuple[float, float],
    contrast_threshold: float,
    noise_threshold: float,
) -> Dict:
  
    img = load_image(path)

    blur_score = compute_blur_score(img)
    brightness, contrast = compute_brightness_contrast(img)
    noise_level = estimate_noise_level(img)

    is_blurry = blur_score < blur_threshold
    is_too_dark = brightness < brightness_range[0]
    is_too_bright = brightness > brightness_range[1]
    is_low_contrast = contrast < contrast_threshold
    is_noisy = noise_level > noise_threshold

    is_low_quality = any(
        [is_blurry, is_too_dark, is_too_bright, is_low_contrast, is_noisy]
    )

    return {
        "path": str(path),
        "blur_score": blur_score,
        "brightness": brightness,
        "contrast": contrast,
        "noise_level": noise_level,
        "is_blurry": is_blurry,
        "is_too_dark": is_too_dark,
        "is_too_bright": is_too_bright,
        "is_low_contrast": is_low_contrast,
        "is_noisy": is_noisy,
        "is_low_quality": is_low_quality,
    }


def list_images_in_dir(root_dir: Path) -> List[Path]:
  
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return [p for p in root_dir.rglob("*") if p.suffix.lower() in exts]


def scan_dataset(
    data_root: str = "data",
    splits: Optional[List[str]] = None,
    blur_threshold: float = 50.0,
    brightness_range: Tuple[float, float] = (60.0, 200.0),
    contrast_threshold: float = 25.0,
    noise_threshold: float = 15.0,
) -> pd.DataFrame:
  
    if splits is None:
        splits = ["train"]  # mostly care about training quality

    data_root = Path(data_root)
    records = []

    for split in splits:
        split_dir = data_root / split
        if not split_dir.exists():
            print(f"[WARN] Split folder not found: {split_dir}")
            continue

        image_paths = list_images_in_dir(split_dir)
        print(f"[INFO] Found {len(image_paths)} images in {split_dir}")

        for path in image_paths:
            info = evaluate_image(
                path=path,
                blur_threshold=blur_threshold,
                brightness_range=brightness_range,
                contrast_threshold=contrast_threshold,
                noise_threshold=noise_threshold,
            )
            info["split"] = split
            # Optional: class name from parent folder inside split
            try:
                info["class"] = path.parent.name
            except Exception:
                info["class"] = "unknown"

            records.append(info)

    df = pd.DataFrame(records)
    return df


def save_quality_report(
    df: pd.DataFrame, output_path: str = "quality_report.csv"
) -> None:
   
    df.to_csv(output_path, index=False)
    print(f"[INFO] Saved quality report to {output_path}")
