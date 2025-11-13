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

import matplotlib.pyplot as plt
import pandas as pd

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="Nodule Detector — Dark Spot Heuristic", layout="wide")
st.title("Nodule Detector — Prototype")

# -----------------------------
# Helpers for the abundance card
# -----------------------------
def choose_size_bin(med_mm: float):
    """Return (label, k) given median nodule size in mm."""
    if med_mm == 0:
        return "Node ? : unknown size", 0.0
    # bins based on your example labels
    if 5 <= med_mm <= 15:
        return "Node 1 : 5–15 mm nodules", st.session_state.get("k_small", 0.50)
    if 25 <= med_mm <= 50:
        return "Node 6 : 25–50 mm nodules", st.session_state.get("k_med", 0.42)
    if 40 <= med_mm <= 90:
        return "Node 7 : 40–90 mm nodules", st.session_state.get("k_large", 0.97)
    # fallback to nearest bucket
    if med_mm < 20:
        return "Node 1 : 5–15 mm nodules*", st.session_state.get("k_small", 0.50)
    if med_mm < 60:
        return "Node 6 : 25–50 mm nodules*", st.session_state.get("k_med", 0.42)
    return "Node 7 : 40–90 mm nodules*", st.session_state.get("k_large", 0.97)

def bin_name_for_diameter(mm: float) -> str:
    if 5 <= mm <= 15:
        return "Node 1 (5–15 mm)"
    if 25 <= mm <= 50:
        return "Node 6 (25–50 mm)"
    if 40 <= mm <= 90:
        return "Node 7 (40–90 mm)"
    return "Other"

# -----------------------------
# Sidebar
# -----------------------------
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

    st.header("Preliminary abundance report")
    mm_per_px = st.number_input("Scale (mm per pixel)", value=0.50, min_value=0.01, step=0.01)
    k_small = st.number_input("k for 5–15 mm (kg/m² per 1% cov)", value=0.50, step=0.05)
    k_med   = st.number_input("k for 25–50 mm", value=0.42, step=0.05)
    k_large = st.number_input("k for 40–90 mm", value=0.97, step=0.05)
    st.session_state["k_small"] = k_small
    st.session_state["k_med"] = k_med
    st.session_state["k_large"] = k_large

    st.header("Patch-classifier (optional)")
    weights_path = st.text_input("Weights path", value="models/patchnet.pt")
    patch = st.number_input("Patch size", 64, 512, 256, step=32)
    stride = st.number_input("Stride", 16, 512, 256, step=16)
    conf_thresh = st.slider("Model confidence threshold", 0.0, 1.0, 0.6, 0.01)

# -----------------------------
# Upload
# -----------------------------
uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "tif", "tiff"])
if not uploaded:
    st.info("Upload an image to begin.")
    st.stop()

# Load image → RGB uint8
img = read_image(uploaded.read())
st.subheader("Original")
st.image(img, caption=f"Loaded image — shape: {img.shape}")

# Decide path
use_heuristic = force_heur or (not Path(weights_path).exists())

# -----------------------------
# Heuristic path
# -----------------------------
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

    # Overlay (neutral, no color highlights)
    overlay = img.copy()
    for (y1, x1, y2, x2) in boxes:
        cy, cx = (y1 + y2) // 2, (x1 + x2) // 2
        r = int(0.5 * max(y2 - y1, x2 - x1))
        cv2.circle(overlay, (cx, cy), r, (255, 255, 255), 2)

    # Coverage & abundance (preliminary)
    h, w = img.shape[:2]
    img_area = float(h * w)
    areas_px = [(y2 - y1) * (x2 - x1) for (y1, x1, y2, x2) in boxes]
    dark_area = float(sum(areas_px))
    coverage_pct = 100.0 * dark_area / img_area if img_area > 0 else 0.0

    diams_px = [max(y2 - y1, x2 - x1) for (y1, x1, y2, x2) in boxes]
    diams_mm = [d * mm_per_px for d in diams_px] if diams_px else []
    med_mm = float(np.median(diams_mm)) if diams_mm else 0.0

    node_label, k = choose_size_bin(med_mm)
    abundance = k * coverage_pct  # kg/m² (linear-in-coverage model; preliminary)

    # UI metrics
    st.subheader("Simple dark-spot detections")
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        st.write(f"**{node_label}**")
        st.caption("Deep Sea Vision Preliminary Report — contains estimates of abundance")
    with c2:
        st.metric("Coverage", f"{coverage_pct:.2f}%")
    with c3:
        st.metric("Abundance est.", f"{abundance:.1f} kg/m²")

    st.image(
        overlay,
        caption=f"Detections: {len(boxes)} — median size {med_mm:.1f} mm",
    )
    with st.expander("Show dark-mask (binary)"):
        st.image(mask)

    # -----------------------------
    # Visualizations & downloads
    # -----------------------------
    st.subheader("Visualizations")

    # Per-detection dataframe (y1,x1,y2,x2, area_px, diameter_px, diameter_mm)
    df = pd.DataFrame([{
        "y1": int(y1), "x1": int(x1), "y2": int(y2), "x2": int(x2),
        "area_px": int((y2 - y1) * (x2 - x1)),
        "diameter_px": float(max(y2 - y1, x2 - x1)),
        "diameter_mm": float(max(y2 - y1, x2 - x1)) * mm_per_px,
        "size_bin": bin_name_for_diameter(float(max(y2 - y1, x2 - x1)) * mm_per_px)
    } for (y1, x1, y2, x2) in boxes])

    if len(df) > 0:
        # Histogram of diameters (mm)
        fig1 = plt.figure()
        plt.hist(df["diameter_mm"].values, bins=20)
        plt.xlabel("Diameter (mm)")
        plt.ylabel("Count")
        plt.title("Size Distribution (diameter in mm)")
        st.pyplot(fig1)

        # Empirical CDF of diameters (mm)
        vals = np.sort(df["diameter_mm"].values)
        ecdf = np.arange(1, len(vals) + 1) / len(vals)
        fig2 = plt.figure()
        plt.step(vals, ecdf, where="post")
        plt.xlabel("Diameter (mm)")
        plt.ylabel("Fraction ≤ x")
        plt.title("Empirical CDF of Nodule Size")
        st.pyplot(fig2)

        # Counts per size bin
        bin_counts = df["size_bin"].value_counts().sort_index()
        fig3 = plt.figure()
        plt.bar(bin_counts.index, bin_counts.values)
        plt.xticks(rotation=20, ha="right")
        plt.ylabel("Count")
        plt.title("Counts per Size Bin")
        st.pyplot(fig3)

        # Download CSV of detections
        st.download_button(
            "Download detections as CSV",
            df.to_csv(index=False).encode("utf-8"),
            file_name="detections.csv",
            mime="text/csv",
        )

        with st.expander("Show detections table"):
            st.dataframe(df, use_container_width=True)
    else:
        st.info("No detections to visualize.")

    # Details table
    st.subheader("Summary")
    st.table({
        "detections": [len(boxes)],
        "median size (mm)": [round(med_mm, 2)],
        "coverage (%)": [round(coverage_pct, 2)],
        "abundance (kg/m²)": [round(abundance, 2)],
    })
    st.stop()

# -----------------------------
# Patch-classifier path
# -----------------------------
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

# simple coverage from patch hits (coarse)
img_area = img.shape[0] * img.shape[1]
covered_area = hits * (int(patch) ** 2)
coverage_pct_model = 100.0 * covered_area / img_area if img_area > 0 else 0.0
st.caption(f"Approx. patch coverage: {coverage_pct_model:.2f}% (coarse)")

top_idx = np.argsort(-nodule_conf)[:5]
st.table({
    "y": [int(coords[i][0]) for i in top_idx],
    "x": [int(coords[i][1]) for i in top_idx],
    "conf": [float(nodule_conf[i]) for i in top_idx],
})
