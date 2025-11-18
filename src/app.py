# src/app.py
from pathlib import Path
import sys

# --- Make sure imports work both locally and in Streamlit Cloud ---
THIS_FILE = Path(__file__).resolve()
SRC_DIR = THIS_FILE.parent           # .../project/src
ROOT_DIR = SRC_DIR.parent            # .../project

for p in (str(ROOT_DIR), str(SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

# -----------------------------
# Imports
# -----------------------------
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Try to import preprocess with robust fallbacks
read_image = None
detect_dark_spots_simple = None
_import_err = None
try:
    from preprocess import read_image, detect_dark_spots_simple  # preferred
except Exception as e1:
    _import_err = e1
    try:
        # alternate name in case your file uses detect_dark_blobs
        from preprocess import read_image, detect_dark_blobs as detect_dark_spots_simple
    except Exception as e2:
        _import_err = (e1, e2)

# If still missing, provide minimal fallbacks so the app can boot
if read_image is None:
    def read_image(file_bytes: bytes) -> np.ndarray:
        """Fallback: decode bytes to RGB uint8 for display and processing."""
        arr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("Failed to decode image bytes (is it a valid image?)")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

if detect_dark_spots_simple is None:
    # very simple dark-spot heuristic (replace with your real one once import is fixed)
    def detect_dark_spots_simple(
        img_rgb: np.ndarray,
        blur_radius: int = 35,
        dark_thresh: int = 10,
        min_area: int = 15,
        max_area: int = 5000,
        min_circularity: float = 0.25,
        sat_thresh: int = 40,
    ):
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(img_hsv)
        if sat_thresh > 0:
            # drop colorful pixels
            v = np.where(s > sat_thresh, 255, v).astype(np.uint8)

        k = (blur_radius | 1)
        g = cv2.GaussianBlur(v, (k, k), 0)
        # dark threshold
        _, thr = cv2.threshold(g, dark_thresh, 255, cv2.THRESH_BINARY_INV)
        thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

        cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in cnts:
            a = cv2.contourArea(c)
            if a < min_area or a > max_area:
                continue
            per = cv2.arcLength(c, True)
            circ = 0 if per == 0 else (4 * np.pi * a) / (per * per)
            if circ < min_circularity:
                continue
            x, y, w, h = cv2.boundingRect(c)  # (x, y, w, h)
            boxes.append((y, x, y + h, x + w))  # (y1, x1, y2, x2)
        return boxes, thr

# Classifier bits (optional)
try:
    from patchify import extract_patches
    from model import load_model
    from infer import predict_patches
except Exception:
    # Allow app to run in heuristic-only mode if these are missing
    extract_patches = load_model = predict_patches = None

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="Nodule Detector — Multi-Image Prototype", layout="wide")
st.title("Nodule Detector — Multi-Image Prototype")

# -----------------------------
# Helpers
# -----------------------------
def choose_size_bin(med_mm: float):
    """Given a median diameter in mm, return (node_label, k)."""
    if med_mm == 0:
        return "Node ? : unknown size", 0.0
    if 5 <= med_mm <= 15:
        return "Node 1 : 5–15 mm nodules", st.session_state.get("k_small", 0.50)
    if 25 <= med_mm <= 50:
        return "Node 6 : 25–50 mm nodules", st.session_state.get("k_med", 0.42)
    if 40 <= med_mm <= 90:
        return "Node 7 : 40–90 mm nodules", st.session_state.get("k_large", 0.97)
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
    sat_thresh = st.slider(
        "Color exclusion (sat)", 0, 255, 40, step=5,
        help="Higher = ignore more colorful pixels; 0 disables color exclusion",
    )

    st.header("Abundance parameters")
    mm_per_px = st.number_input("Scale (mm per pixel)", value=0.50, min_value=0.01, step=0.01)
    k_small = st.number_input("k (5–15 mm)", value=0.50, step=0.05)
    k_med   = st.number_input("k (25–50 mm)", value=0.42, step=0.05)
    k_large = st.number_input("k (40–90 mm)", value=0.97, step=0.05)
    st.session_state["k_small"] = k_small
    st.session_state["k_med"] = k_med
    st.session_state["k_large"] = k_large

    st.header("Patch-classifier (optional)")
    weights_path = st.text_input("Weights path", value="models/patchnet.pt")
    patch = st.number_input("Patch size", 64, 512, 256, step=32)
    stride = st.number_input("Stride", 16, 512, 256, step=16)
    conf_thresh = st.slider("Model confidence threshold", 0.0, 1.0, 0.6, 0.01)

# -----------------------------
# Upload multiple images
# -----------------------------
uploaded_files = st.file_uploader(
    "Upload one or more seafloor images",
    type=["jpg", "jpeg", "png", "tif", "tiff"],
    accept_multiple_files=True,
)
if not uploaded_files:
    if _import_err:
        st.warning(f"Note: had trouble importing from preprocess: {_import_err}")
    st.info("Upload at least one image to begin.")
    st.stop()

# Decide path
use_heuristic = (
    force_heur
    or weights_path.strip() == ""
    or not Path(weights_path).exists()
    or load_model is None
)

# --- collectors for aggregate stats across all uploads ---
all_det_rows = []          # per-detection rows from every image
total_img_area = 0.0       # sum of image areas (px)
total_dark_area = 0.0      # sum of detected dark area (px²)
per_image_summ = []        # per-image summary rows

# -----------------------------
# Loop over images
# -----------------------------
for uploaded in uploaded_files:
    # Read image bytes and decode
    img = read_image(uploaded.read())
    st.markdown(f"### {uploaded.name}")
    st.image(img, caption=f"Loaded — shape: {img.shape}", width=600)

    if use_heuristic:
        # Heuristic detector
        boxes, mask = detect_dark_spots_simple(
            img_rgb=img,
            blur_radius=int(blur_radius),
            dark_thresh=int(dark_thresh),
            min_area=int(min_area),
            max_area=int(max_area),
            min_circularity=float(min_circ),
            sat_thresh=int(sat_thresh),
        )

        # Overlay with neutral (white) circles
        overlay = img.copy()
        for (y1, x1, y2, x2) in boxes:
            cy, cx = (y1 + y2) // 2, (x1 + x2) // 2
            r = int(0.5 * max(y2 - y1, x2 - x1))
            cv2.circle(overlay, (cx, cy), r, (255, 255, 255), 2)

        # Coverage & abundance
        h, w = img.shape[:2]
        img_area = float(h * w)
        areas_px = [(y2 - y1) * (x2 - x1) for (y1, x1, y2, x2) in boxes]
        dark_area = float(sum(areas_px))
        coverage_pct = 100.0 * dark_area / img_area if img_area > 0 else 0.0

        diams_px = [max(y2 - y1, x2 - x1) for (y1, x1, y2, x2) in boxes]
        diams_mm = [d * mm_per_px for d in diams_px] if diams_px else []
        med_mm = float(np.median(diams_mm)) if diams_mm else 0.0

        node_label, k = choose_size_bin(med_mm)
        abundance = k * coverage_pct  # kg/m² (simple linear model)

        # --- aggregate collectors ---
        total_img_area += img_area
        total_dark_area += dark_area
        per_image_summ.append({
            "image": uploaded.name,
            "detections": len(boxes),
            "median_size_mm": float(med_mm),
            "coverage_pct": float(coverage_pct),
            "abundance_kg_m2": float(abundance),
        })
        if len(boxes) > 0:
            for (y1, x1, y2, x2) in boxes:
                area_px = int((y2 - y1) * (x2 - x1))
                diam_mm = float(max(y2 - y1, x2 - x1)) * mm_per_px
                all_det_rows.append({
                    "image": uploaded.name,
                    "y1": int(y1),
                    "x1": int(x1),
                    "y2": int(y2),
                    "x2": int(x2),
                    "area_px": area_px,
                    "diameter_mm": diam_mm,
                    "size_bin": bin_name_for_diameter(diam_mm),
                })

        # --- per-image UI ---
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
            width=600,
        )
        with st.expander("Dark-mask (binary)"):
            st.image(mask, width=600)

        # Per-image visualizations
        if len(boxes) > 0:
            df = pd.DataFrame([{
                "y1": int(y1),
                "x1": int(x1),
                "y2": int(y2),
                "x2": int(x2),
                "area_px": int((y2 - y1) * (x2 - x1)),
                "diameter_mm": float(max(y2 - y1, x2 - x1)) * mm_per_px,
                "size_bin": bin_name_for_diameter(float(max(y2 - y1, x2 - x1)) * mm_per_px),
            } for (y1, x1, y2, x2) in boxes])

            with st.expander("Visualizations", expanded=False):
                # histogram
                fig1, ax1 = plt.subplots(figsize=(5, 3))
                ax1.hist(df["diameter_mm"], bins=20)
                ax1.set_xlabel("Diameter (mm)")
                ax1.set_ylabel("Count")
                ax1.set_title("Size Distribution")
                st.pyplot(fig1)

                # ECDF
                vals = np.sort(df["diameter_mm"].values)
                ecdf = np.arange(1, len(vals) + 1) / len(vals)
                fig2, ax2 = plt.subplots(figsize=(5, 3))
                if len(vals) > 0:
                    ax2.step(vals, ecdf, where="post")
                ax2.set_xlabel("Diameter (mm)")
                ax2.set_ylabel("Fraction ≤ x")
                ax2.set_title("Empirical CDF")
                st.pyplot(fig2)

                # size-bin bar
                bin_counts = df["size_bin"].value_counts().sort_index()
                fig3, ax3 = plt.subplots(figsize=(5, 3))
                ax3.bar(bin_counts.index, bin_counts.values)
                ax3.set_ylabel("Count")
                ax3.set_title("Counts per Size Bin")
                plt.xticks(rotation=15, ha="right")
                st.pyplot(fig3)

                st.download_button(
                    "Download detections for this image (CSV)",
                    df.to_csv(index=False).encode("utf-8"),
                    file_name=f"detections_{uploaded.name}.csv",
                    mime="text/csv",
                )

        st.divider()
        st.table({
            "detections": [len(boxes)],
            "median size (mm)": [round(med_mm, 2)],
            "coverage (%)": [round(coverage_pct, 2)],
            "abundance (kg/m²)": [round(abundance, 2)],
        })

    else:
        # Classifier path (optional, if you want it)
        if extract_patches is None or load_model is None or predict_patches is None:
            st.error("Patch-classifier modules not available. Enable heuristic mode or add model files.")
            continue

        patches, coords = extract_patches(img, size=int(patch), stride=int(stride))
        try:
            model, device = load_model(weights_path)
        except Exception as e:
            st.error(f"Failed to load weights at '{weights_path}': {e}")
            continue

        labels, probs = predict_patches(model, device, patches)
        nodule_conf = probs[:, 1] if probs.shape[1] >= 2 else np.zeros(len(patches), dtype=np.float32)

        overlay = img.copy().astype(np.uint8)
        p_sz, hits = int(patch), 0
        for (y, x), conf in zip(coords, nodule_conf):
            if conf >= conf_thresh:
                hits += 1
                cv2.rectangle(
                    overlay,
                    (x, y),
                    (min(x + p_sz, overlay.shape[1]), min(y + p_sz, overlay.shape[0])),
                    (255, 255, 255),
                    2,
                )

        st.image(overlay, caption=f"Patches ≥ {conf_thresh}: {hits}", width=600)

# =============================
# Aggregate across all uploads
# =============================
st.header("Aggregate across uploads")

if len(all_det_rows) == 0:
    st.info("No detections across all images to aggregate.")
else:
    agg_df = pd.DataFrame(all_det_rows)

    # Global plots
    col1, col2 = st.columns(2)
    with col1:
        fig1, ax1 = plt.subplots(figsize=(5, 3))
        ax1.hist(agg_df["diameter_mm"], bins=30)
        ax1.set_xlabel("Diameter (mm)")
        ax1.set_ylabel("Count")
        ax1.set_title("All Images — Size Distribution")
        st.pyplot(fig1)

    with col2:
        vals = np.sort(agg_df["diameter_mm"].values)
        ecdf = np.arange(1, len(vals) + 1) / len(vals) if len(vals) > 0 else np.array([])
        fig2, ax2 = plt.subplots(figsize=(5, 3))
        if len(vals) > 0:
            ax2.step(vals, ecdf, where="post")
        ax2.set_xlabel("Diameter (mm)")
        ax2.set_ylabel("Fraction ≤ x")
        ax2.set_title("All Images — Empirical CDF")
        st.pyplot(fig2)

    # Size-bin bar chart
    bin_counts = agg_df["size_bin"].value_counts().sort_index()
    fig3, ax3 = plt.subplots(figsize=(6, 3))
    ax3.bar(bin_counts.index, bin_counts.values)
    ax3.set_ylabel("Count")
    ax3.set_title("All Images — Counts per Size Bin")
    plt.xticks(rotation=15, ha="right")
    st.pyplot(fig3)

    # Pooled coverage = sum(dark areas) / sum(image areas)
    pooled_cov = 100.0 * total_dark_area / total_img_area if total_img_area > 0 else 0.0
    global_med = float(agg_df["diameter_mm"].median()) if len(agg_df) else 0.0
    node_label, k_global = choose_size_bin(global_med)
    pooled_abundance = k_global * pooled_cov

    st.subheader("Aggregate summary")
    st.table({
        "images": [len(per_image_summ)],
        "total detections": [len(agg_df)],
        "global median size (mm)": [round(global_med, 2)],
        "pooled coverage (%)": [round(pooled_cov, 2)],
        "pooled abundance (kg/m²)": [round(pooled_abundance, 2)],
    })

    st.download_button(
        "Download ALL detections as CSV",
        agg_df.to_csv(index=False).encode("utf-8"),
        file_name="all_detections.csv",
        mime="text/csv",
    )

    with st.expander("Per-image summary"):
        st.dataframe(pd.DataFrame(per_image_summ), use_container_width=True)
