# src/app.py
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import streamlit as st

from preprocess import read_image, detect_dark_spots_simple
from patchify import extract_patches
from model import load_model
from infer import predict_patches

st.set_page_config(page_title="Nodule Detector — Dark Spot Heuristic", layout="wide")
st.title("Nodule Detector — Prototype")

with st.sidebar:
    st.header("Heuristic (dark spots on gray)")
    force_heur = st.checkbox("Force heuristic (ignore weights)", value=True)
    blur_radius = st.slider("Background blur radius (odd)", 9, 99, 35, step=2)
    dark_thresh = st.slider("Darkness threshold (0–255)", 1, 40, 10, step=1)
    min_area = st.slider("Min blob area (px²)", 10, 200, 15, step=5)
    max_area = st.slider("Max blob area (px²)", 500, 20000, 5000, step=500)
    min_circ = st.slider("Min circularity", 0.10, 0.80, 0.25, step=0.05)
    sat_thresh = st.slider("Color exclusion (sat)", 0, 255, 40, step=5,
                           help="Higher = ignore more colorful pixels; 0 disables color exclusion")

    st.header("Patch-classifier (optional)")
    weights_path = st.text_input("Weights path", value="models/patchnet.pt")
    patch = st.number_input("Patch size", 64, 512, 256, step=32)
    stride = st.number_input("Stride", 16, 512, 256, step=16)
    conf_thresh = st.slider("Model confidence threshold", 0.0, 1.0, 0.6, 0.01)

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "tif", "tiff"])
if not uploaded:
    st.info("Upload an image to begin.")
    st.stop()

# Load image → RGB uint8
img = read_image(uploaded.read())
st.subheader("Original")
st.image(img, caption=f"Loaded image — shape: {img.shape}")

# Choose path
use_heuristic = force_heur or (not Path(weights_path).exists())

if use_heuristic:
    # Dark-spot detector with color exclusion
    boxes, mask = detect_dark_spots_simple(
        img_rgb=img,
        blur_radius=int(blur_radius),
        dark_thresh=int(dark_thresh),
        min_area=int(min_area),
        max_area=int(max_area),
        min_circularity=float(min_circ),
        sat_thresh=int(sat_thresh),
    )

    overlay = img.copy()
    # draw neutral (white) circles to avoid colored highlights
    for (y1, x1, y2, x2) in boxes:
        cy, cx = (y1 + y2) // 2, (x1 + x2) // 2
        r = int(0.5 * max(y2 - y1, x2 - x1))
        cv2.circle(overlay, (cx, cy), r, (255, 255, 255), 2)

    st.subheader("Simple dark-spot detections")
    st.image(overlay, caption=f"Detections: {len(boxes)}")
    with st.expander("Show dark-mask (binary)"):
        st.image(mask)
    st.stop()

# Patch-classifier path (if weights present and not forcing heuristic)
patches, coords = extract_patches(img, size=int(patch), stride=int(stride))
try:
    model, device = load_model(weights_path)
except Exception as e:
    st.error(f"Failed to load weights at '{weights_path}': {e}")
    st.stop()

labels, probs = predict_patches(model, device, patches)
nodule_conf = probs[:, 1] if probs.shape[1] >= 2 else np.zeros(len(patches), dtype=np.float32)

overlay = img.copy().astype(np.uint8)
p = int(patch)
hits = 0
for (y, x), conf in zip(coords, nodule_conf):
    if conf >= conf_thresh:
        hits += 1
        # neutral (white) rectangles
        cv2.rectangle(
            overlay,
            (x, y),
            (min(x + p, overlay.shape[1]), min(y + p, overlay.shape[0])),
            (255, 255, 255),
            2,
        )

st.subheader("Model detections")
st.image(overlay, caption=f"Patches above threshold: {hits}")

top_idx = np.argsort(-nodule_conf)[:5]
st.table({
    "y": [int(coords[i][0]) for i in top_idx],
    "x": [int(coords[i][1]) for i in top_idx],
    "conf": [float(nodule_conf[i]) for i in top_idx],
})
