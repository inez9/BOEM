# src/preprocess.py
import io
from typing import List, Tuple
import numpy as np
from PIL import Image
import tifffile as tiff
import cv2


# ─────────────────────────────────────────────────────────────
# 1. Robust image loader (TIFF/JPG/PNG safe)
# ─────────────────────────────────────────────────────────────
def read_image(file_bytes: bytes) -> np.ndarray:
    """
    Load a .tif, .png, or .jpg file (bytes) into an RGB uint8 numpy array.
    Converts grayscale or multi-band images to 3-channel RGB.
    """
    try:
        with tiff.TiffFile(io.BytesIO(file_bytes)) as tf:
            arr = tf.asarray()
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            elif arr.ndim == 3 and arr.shape[-1] not in (3, 4):
                arr = np.stack([arr[..., 0]] * 3, axis=-1)
            arr = arr.astype(np.float32)
            arr = 255 * (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
            return arr.astype(np.uint8)
    except Exception:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        return np.array(img)

def detect_dark_spots_simple(
    img_rgb: np.ndarray,
    blur_radius: int = 31,
    dark_thresh: int = 12,
    min_area: int = 25,
    max_area: int = 5000,
    min_circularity: float = 0.25,
    sat_thresh: int = 40,   # new param: exclude colorful regions
) -> Tuple[List[Tuple[int, int, int, int]], np.ndarray]:
    """
    Detect darker blobs on uniform gray background, ignoring colored areas.

    Steps:
      1) Remove high-saturation (colored) regions
      2) Convert to gray and blur for background
      3) Subtract background → highlight dark spots
      4) Threshold and clean mask
      5) Filter blobs by area & circularity
    """

    # 1️⃣ Mask out colored regions (exclude vivid reds/blues)
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    s_channel = hsv[:, :, 1]            # saturation channel
    neutral_mask = s_channel < sat_thresh  # True where nearly gray
    img_neutral = img_rgb.copy()
    img_neutral[~neutral_mask] = 255     # paint non-neutral areas white

    # 2️⃣ Detect dark spots on neutral-only image
    gray = cv2.cvtColor(img_neutral, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (blur_radius, blur_radius), 0)
    diff = cv2.subtract(blur, gray)  # bright = darker-than-background
    _, mask = cv2.threshold(diff, dark_thresh, 255, cv2.THRESH_BINARY)

    # 3️⃣ Clean small specks
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

    # 4️⃣ Find contours and filter
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: List[Tuple[int, int, int, int]] = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        per = cv2.arcLength(c, True)
        circ = 0 if per == 0 else (4 * np.pi * area) / (per**2)
        if circ < min_circularity:
            continue
        x, y, w, h = cv2.boundingRect(c)
        boxes.append((y, x, y + h, x + w))

    return boxes, mask
