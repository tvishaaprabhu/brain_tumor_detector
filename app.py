import streamlit as st
import numpy as np
from PIL import Image
import cv2
import io
import pydicom
import pandas as pd

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroScan AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── Global CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* Reset & base */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0A0E1A;
    color: #E8EAF0;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem 3rem; max-width: 1100px; }

/* Hero */
.hero {
    text-align: center;
    padding: 3rem 0 2rem 0;
    border-bottom: 1px solid #1E2433;
    margin-bottom: 2.5rem;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.8rem;
    font-weight: 700;
    color: #FFFFFF;
    letter-spacing: -0.03em;
    margin: 0;
}
.hero-title span { color: #4ECDC4; }
.hero-sub {
    font-size: 0.95rem;
    color: #6B7280;
    margin-top: 0.5rem;
    font-weight: 300;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

/* Section headers */
.section-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #4ECDC4;
    margin-bottom: 0.4rem;
}
.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.4rem;
    font-weight: 600;
    color: #FFFFFF;
    margin-bottom: 1.2rem;
}

/* Cards */
.card {
    background: #111827;
    border: 1px solid #1E2433;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

/* Result card */
.result-card {
    background: linear-gradient(135deg, #111827 0%, #0D1520 100%);
    border: 1px solid #4ECDC4;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin: 1rem 0;
}
.result-label {
    font-size: 0.75rem;
    color: #6B7280;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.5rem;
}
.result-tumor {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: #FFFFFF;
    margin-bottom: 0.3rem;
}
.result-confidence {
    font-size: 1rem;
    color: #4ECDC4;
    font-weight: 500;
}
.result-clear {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: #4ECDC4;
}

/* Probability bars */
.prob-row {
    display: flex;
    align-items: center;
    margin-bottom: 0.6rem;
    gap: 0.8rem;
}
.prob-label {
    font-size: 0.8rem;
    color: #9CA3AF;
    width: 120px;
    flex-shrink: 0;
}
.prob-bar-bg {
    flex: 1;
    background: #1E2433;
    border-radius: 4px;
    height: 6px;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, #4ECDC4, #45B7D1);
    transition: width 0.4s ease;
}
.prob-pct {
    font-size: 0.78rem;
    color: #6B7280;
    width: 38px;
    text-align: right;
    flex-shrink: 0;
}

/* Upload zone */
.upload-hint {
    font-size: 0.82rem;
    color: #6B7280;
    margin-top: 0.5rem;
}

/* Divider */
.divider {
    border: none;
    border-top: 1px solid #1E2433;
    margin: 2rem 0;
}

/* Toggle row */
.toggle-row {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
    margin-bottom: 0.5rem;
}

/* Streamlit widget overrides */
.stCheckbox label { color: #9CA3AF !important; font-size: 0.85rem !important; }
.stSlider label { color: #9CA3AF !important; font-size: 0.85rem !important; }
.stButton > button {
    background: #4ECDC4 !important;
    color: #0A0E1A !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 1.8rem !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.02em !important;
}
.stButton > button:hover {
    background: #45B7D1 !important;
}
[data-testid="stFileUploader"] {
    border: 2px dashed #1E2433 !important;
    border-radius: 12px !important;
    background: #111827 !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Hero ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <p class="hero-title">Neuro<span>Scan</span> AI</p>
    <p class="hero-sub">Brain Tumor Detection &amp; Classification · Powered by Deep Learning</p>
</div>
""", unsafe_allow_html=True)

# ─── Step 1: Upload ───────────────────────────────────────────────────────────
st.markdown('<p class="section-label">Step 01</p><p class="section-title">Upload MRI Scan</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Drop your MRI scan here",
    type=["jpg", "jpeg", "png", "webp", "dcm"],
    label_visibility="collapsed"
)
st.markdown('<p class="upload-hint">Supports JPG, PNG, WebP, and DICOM (.dcm) files</p>', unsafe_allow_html=True)

if uploaded_file is not None:
    is_dicom = uploaded_file.name.lower().endswith(".dcm")

    if is_dicom:
        dicom = pydicom.dcmread(uploaded_file)
        pixel_array = dicom.pixel_array.squeeze()

        if len(pixel_array.shape) == 3:
            n_slices = pixel_array.shape[0]
            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            st.markdown('<p class="section-label">Multi-Slice DICOM</p>', unsafe_allow_html=True)
            st.caption(f"{n_slices} slices detected — use the controls below to select a slice.")

            if "slice_idx" not in st.session_state:
                st.session_state.slice_idx = 0

            col_prev, col_slider, col_next = st.columns([1, 10, 1])
            with col_prev:
                st.write("")
                if st.button("◀"):
                    st.session_state.slice_idx = max(0, st.session_state.slice_idx - 1)
            with col_slider:
                st.session_state.slice_idx = st.slider(
                    "Slice", 0, n_slices - 1, st.session_state.slice_idx,
                    format="Slice %d"
                )
            with col_next:
                st.write("")
                if st.button("▶"):
                    st.session_state.slice_idx = min(n_slices - 1, st.session_state.slice_idx + 1)

            st.caption(f"Viewing slice {st.session_state.slice_idx + 1} of {n_slices}")
            img_array = pixel_array[st.session_state.slice_idx]
        else:
            img_array = pixel_array

        img_array = cv2.normalize(img_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    else:
        img = Image.open(uploaded_file)
        img_array = np.array(img.convert("L")).squeeze()

    # Show uploaded scan
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    col_img, col_info = st.columns([1, 1])
    with col_img:
        st.image(img_array, caption="Uploaded scan", use_container_width=True)
    with col_info:
        h, w = img_array.shape[:2]
        st.markdown(f"""
        <div class="card">
            <p class="section-label">Scan Info</p>
            <p style="color:#9CA3AF;font-size:0.85rem;margin:0.2rem 0;">
                <b style="color:#E8EAF0;">File</b>&nbsp;&nbsp;{uploaded_file.name}
            </p>
            <p style="color:#9CA3AF;font-size:0.85rem;margin:0.2rem 0;">
                <b style="color:#E8EAF0;">Size</b>&nbsp;&nbsp;{w} × {h} px
            </p>
            <p style="color:#9CA3AF;font-size:0.85rem;margin:0.2rem 0;">
                <b style="color:#E8EAF0;">Format</b>&nbsp;&nbsp;{"DICOM" if is_dicom else uploaded_file.type}
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ─── Step 2: Preprocessing ────────────────────────────────────────────────
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<p class="section-label">Step 02</p><p class="section-title">Image Preprocessing</p>', unsafe_allow_html=True)

    auto_mode = st.toggle("Auto preprocessing", value=True)

    if auto_mode:
        preprocessed = cv2.normalize(img_array.copy(), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        processed_eq = preprocessed.copy()
        mask_eq = processed_eq > 15
        if mask_eq.any():
            brain_pixels = processed_eq[mask_eq]
            brain_eq = cv2.equalizeHist(brain_pixels.reshape(-1, 1))
            processed_eq[mask_eq] = brain_eq.ravel()
        preprocessed = processed_eq
        denoised = cv2.GaussianBlur(preprocessed, (3, 3), 0)
        st.caption("Auto mode: normalisation + brain-masked histogram equalisation + light Gaussian denoising applied.")
    else:
        st.caption("Manual mode — select your own preprocessing options.")
        col_a, col_b = st.columns(2)
        with col_a:
            do_normalize = st.checkbox("Normalize (0–255)", value=True)
            do_equalize = st.checkbox("Histogram Equalization")
            do_flip = st.checkbox("Horizontal Flip")
        with col_b:
            do_gaussian = st.checkbox("Gaussian Blur")
            do_median = st.checkbox("Median Filter")
            do_nlm = st.checkbox("Non-Local Means")

        preprocessed = img_array.copy()
        if do_normalize:
            preprocessed = cv2.normalize(preprocessed, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if do_equalize:
            processed_eq = preprocessed.copy()
            mask_eq = processed_eq > 15
            if mask_eq.any():
                brain_pixels = processed_eq[mask_eq]
                brain_eq = cv2.equalizeHist(brain_pixels.reshape(-1, 1))
                processed_eq[mask_eq] = brain_eq.ravel()
            preprocessed = processed_eq
        if do_flip:
            preprocessed = cv2.flip(preprocessed, 1)

        denoised = preprocessed.copy()
        if do_gaussian:
            k = st.slider("Gaussian kernel", 1, 15, 5, step=2)
            denoised = cv2.GaussianBlur(denoised, (k, k), 0)
        if do_median:
            k2 = st.slider("Median kernel", 1, 15, 5, step=2)
            denoised = cv2.medianBlur(denoised, k2)
        if do_nlm:
            h_val = st.slider("NLM strength", 1, 30, 10)
            denoised = cv2.fastNlMeansDenoising(denoised, h=h_val)

    col1, col2 = st.columns(2)
    with col1:
        st.image(img_array, caption="Original", use_container_width=True)
    with col2:
        st.image(denoised, caption="Processed", use_container_width=True)

    # ─── Step 3: Diagnosis ────────────────────────────────────────────────────
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<p class="section-label">Step 03</p><p class="section-title">AI Diagnosis</p>', unsafe_allow_html=True)

    run_btn = st.button("Run Diagnosis →")

    if run_btn:
        try:
            from predict import load_model, preprocess, predict, get_gradcam, overlay_gradcam, CLASS_NAMES
            MODEL_PATH = "brain_tumor_detector.keras"
            model = load_model(MODEL_PATH)

            input_tensor = preprocess(denoised)
            gray_128 = cv2.resize(denoised, (128, 128)).astype(np.float32) / 255.0

            with st.spinner("Analysing scan..."):
                class_name, confidence, all_probs = predict(model, input_tensor)

            detected_class = class_name.lower().replace(" ", "")

            # Result card
            if detected_class == "notumor":
                st.markdown(f"""
                <div class="result-card">
                    <p class="result-label">Diagnosis</p>
                    <p class="result-clear">No Tumor Detected</p>
                    <p class="result-confidence">{confidence*100:.1f}% confidence</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card">
                    <p class="result-label">Tumor Detected</p>
                    <p class="result-tumor">{class_name}</p>
                    <p class="result-confidence">{confidence*100:.1f}% confidence</p>
                </div>
                """, unsafe_allow_html=True)

            # Probability breakdown
            st.markdown("<p style='color:#6B7280;font-size:0.8rem;margin-top:1.2rem;margin-bottom:0.6rem;text-transform:uppercase;letter-spacing:0.08em;'>Class probabilities</p>", unsafe_allow_html=True)
            for idx, name in CLASS_NAMES.items():
                pct = float(all_probs[idx]) * 100
                bar_w = f"{pct:.1f}%"
                st.markdown(f"""
                <div class="prob-row">
                    <span class="prob-label">{name}</span>
                    <div class="prob-bar-bg">
                        <div class="prob-bar-fill" style="width:{bar_w};"></div>
                    </div>
                    <span class="prob-pct">{pct:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)

            # Grad-CAM
            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            st.markdown('<p class="section-label">Step 04</p><p class="section-title">Grad-CAM Explanation</p>', unsafe_allow_html=True)
            st.caption("Highlights the regions that most influenced the model's prediction.")

            with st.spinner("Generating Grad-CAM..."):
                class_idx = int(np.argmax(all_probs))
                heatmap = get_gradcam(model, input_tensor, class_idx)
                overlaid = overlay_gradcam(gray_128, heatmap)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(gray_128, caption="Original scan", clamp=True, use_container_width=True)
            with col2:
                st.image(heatmap, caption="Activation heatmap", use_container_width=True)
            with col3:
                st.image(overlaid, caption="Overlay", use_container_width=True)

            # Download report
            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            buf_overlay = io.BytesIO()
            Image.fromarray(overlaid).save(buf_overlay, format="PNG")
            st.download_button(
                label="Download Grad-CAM Report",
                data=buf_overlay.getvalue(),
                file_name=f"gradcam_{uploaded_file.name.rsplit('.', 1)[0]}.png",
                mime="image/png"
            )

        except Exception as e:
            st.error(f"Diagnosis failed: {e}")
            st.info("Make sure `brain_tumor_detector.keras` and `predict.py` are in the repo root.")

else:
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center;padding:3rem 0;color:#2D3748;">
        <p style="font-size:3rem;margin:0;">🧠</p>
        <p style="font-family:'Space Grotesk',sans-serif;font-size:1.1rem;color:#374151;margin-top:0.5rem;">
            Upload a scan to begin
        </p>
    </div>
    """, unsafe_allow_html=True)
