import os
import io
import base64
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
from PIL import Image
import cv2
import pydicom

st.set_page_config(
    page_title="NeuroScan AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,300;0,400;0,500;1,400&family=DM+Serif+Display&family=Space+Mono:wght@400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0c0f1a !important;
    color: #c9cdd9;
}
.stApp, [data-testid="stAppViewContainer"],
[data-testid="stHeader"], [data-testid="stToolbar"],
[data-testid="stMain"] {
    background-color: #0c0f1a !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 4rem 6rem 4rem; max-width: 1200px; }

body {
    background-image:
        radial-gradient(ellipse 60% 40% at 15% -5%, rgba(124,158,255,0.08), transparent 60%),
        radial-gradient(ellipse 60% 40% at 100% 100%, rgba(155,107,255,0.06), transparent 70%);
    background-attachment: fixed;
}

/* Header */
.ns-header {
    display: flex; align-items: flex-start; justify-content: space-between;
    padding: 2.8rem 0; border-bottom: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 3rem;
}
.ns-wordmark {
    font-family: 'DM Serif Display', serif; font-size: 2.2rem;
    color: #f0f2f8; letter-spacing: -0.01em; line-height: 1;
}
.ns-wordmark .scan { color: #7c9eff; }
.ns-tagline {
    font-family: 'Space Mono', monospace; font-size: 0.6rem; color: #3d4460;
    letter-spacing: 0.14em; text-transform: uppercase; line-height: 1.8;
    text-align: right; margin-top: 0.2rem;
}

/* Step headers */
.ns-eyebrow {
    font-family: 'Space Mono', monospace; font-size: 0.6rem;
    letter-spacing: 0.18em; text-transform: uppercase;
    color: #7c9eff; margin-bottom: 0.35rem;
}
.ns-title {
    font-family: 'DM Serif Display', serif; font-size: 1.75rem;
    color: #f0f2f8; letter-spacing: -0.015em; margin-bottom: 1.4rem; line-height: 1.1;
}

/* Upload */
.ns-upload-hint {
    font-family: 'Space Mono', monospace; font-size: 0.6rem; color: #3d4460;
    letter-spacing: 0.1em; text-transform: uppercase; margin-top: 0.5rem;
}
[data-testid="stFileUploader"] section {
    background: linear-gradient(160deg, #131829 0%, #0e1220 100%) !important;
    border: 1px solid rgba(124,158,255,0.12) !important;
    border-radius: 14px !important; padding: 1.5rem !important;
}

/* Info card */
.ns-info-card {
    background: linear-gradient(160deg, #131829 0%, #0e1220 100%);
    border: 1px solid rgba(124,158,255,0.1);
    border-radius: 14px; overflow: hidden;
}
.ns-info-row {
    display: flex; justify-content: space-between; align-items: baseline;
    padding: 0.85rem 1.4rem; border-bottom: 1px solid rgba(255,255,255,0.04);
}
.ns-info-row:last-child { border-bottom: none; }
.ns-info-key {
    font-family: 'Space Mono', monospace; font-size: 0.62rem; color: #3d4460;
    text-transform: uppercase; letter-spacing: 0.12em;
}
.ns-info-val {
    font-size: 0.82rem; color: #c9cdd9; text-align: right;
    max-width: 58%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}

/* Image labels + wrap */
.ns-img-label {
    font-family: 'Space Mono', monospace; font-size: 0.58rem; color: #3d4460;
    text-transform: uppercase; letter-spacing: 0.14em; margin-bottom: 0.5rem;
}
.ns-img-label-active {
    font-family: 'Space Mono', monospace; font-size: 0.58rem; color: #7c9eff;
    text-transform: uppercase; letter-spacing: 0.14em; margin-bottom: 0.5rem;
}
.ns-img-label-active::before { content: "● "; }
.ns-img-wrap {
    background: #090c14; border-radius: 10px; overflow: hidden;
    border: 1px solid rgba(255,255,255,0.04);
}

/* Auto badge */
.ns-auto-badge {
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: rgba(124,158,255,0.07); border: 1px solid rgba(124,158,255,0.18);
    border-radius: 20px; padding: 0.22rem 0.7rem;
    font-family: 'Space Mono', monospace; font-size: 0.58rem; color: #7c9eff;
    letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 0.5rem;
}
.ns-auto-badge::before { content: "●  "; }
.ns-applied {
    font-size: 0.78rem; color: #3d4460; font-style: italic; margin-bottom: 1.2rem;
}
.ns-applied b { color: #8b91a3; font-style: normal; font-weight: 400; }

/* Manual card */
.ns-manual-card {
    background: linear-gradient(160deg, #131829 0%, #0e1220 100%);
    border: 1px solid rgba(124,158,255,0.08);
    border-radius: 12px; padding: 1.2rem 1.6rem; margin-bottom: 1.2rem;
}
.stCheckbox label {
    color: #c9cdd9 !important; font-size: 0.88rem !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #7c9eff 0%, #9b6bff 100%) !important;
    color: #0c0f1a !important; font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important; border: none !important; border-radius: 8px !important;
    padding: 0.6rem 1.6rem !important; font-size: 0.88rem !important;
    letter-spacing: 0.02em !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

[data-testid="stDownloadButton"] button {
    background: rgba(124,158,255,0.08) !important; color: #7c9eff !important;
    border: 1px solid rgba(124,158,255,0.2) !important;
    font-size: 0.84rem !important; border-radius: 8px !important;
    padding: 0.5rem 1.2rem !important;
}

/* Result cards */
.ns-result-tumor {
    position: relative; overflow: hidden;
    background: linear-gradient(160deg, #131829 0%, #160e28 100%);
    border: 1px solid rgba(155,107,255,0.25); border-radius: 16px;
    padding: 3rem 2rem; text-align: center; margin: 1.4rem 0;
}
.ns-result-tumor::before {
    content: ''; position: absolute; top: -80px; right: -80px;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(155,107,255,0.12) 0%, transparent 65%);
    pointer-events: none;
}
.ns-result-clear {
    position: relative; overflow: hidden;
    background: linear-gradient(160deg, #0d1a1a 0%, #0a1520 100%);
    border: 1px solid rgba(100,220,180,0.2); border-radius: 16px;
    padding: 3rem 2rem; text-align: center; margin: 1.4rem 0;
}
.ns-result-clear::before {
    content: ''; position: absolute; top: -80px; right: -80px;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(100,220,180,0.08) 0%, transparent 65%);
    pointer-events: none;
}
.ns-result-eyebrow {
    font-family: 'Space Mono', monospace; font-size: 0.6rem;
    letter-spacing: 0.16em; text-transform: uppercase; margin-bottom: 0.8rem;
}
.ns-eb-tumor { color: #9b6bff; }
.ns-eb-clear { color: #64dcb4; }
.ns-result-name-tumor {
    font-family: 'DM Serif Display', serif; font-size: 3rem; color: #f0f2f8;
    line-height: 1; margin-bottom: 0.6rem; letter-spacing: -0.02em;
}
.ns-result-name-clear {
    font-family: 'DM Serif Display', serif; font-size: 3rem; color: #64dcb4;
    line-height: 1; margin-bottom: 0.6rem; letter-spacing: -0.02em;
}
.ns-conf-tumor {
    font-family: 'Space Mono', monospace; font-size: 0.82rem;
    color: #9b6bff; letter-spacing: 0.06em;
}
.ns-conf-clear {
    font-family: 'Space Mono', monospace; font-size: 0.82rem;
    color: #64dcb4; letter-spacing: 0.06em;
}
.ns-rdiv { width: 200px; height: 1px; background: rgba(155,107,255,0.2); margin: 1.2rem auto; }
.ns-rdiv-clear { width: 200px; height: 1px; background: rgba(100,220,180,0.15); margin: 1.2rem auto; }

/* Metric tiles */
.ns-metrics { display: flex; gap: 0.8rem; margin-top: 1.2rem; }
.ns-metric {
    flex: 1;
    background: linear-gradient(160deg, #131829 0%, #0e1220 100%);
    border: 1px solid rgba(124,158,255,0.1);
    border-radius: 12px; padding: 1.1rem 0.8rem; text-align: center;
}
.ns-metric-key {
    font-family: 'Space Mono', monospace; font-size: 0.55rem; color: #3d4460;
    text-transform: uppercase; letter-spacing: 0.14em; margin-bottom: 0.4rem;
}
.ns-metric-val {
    font-family: 'DM Serif Display', serif; font-size: 1.5rem; color: #f0f2f8;
    line-height: 1;
}
.ns-metric-unit {
    font-family: 'Space Mono', monospace; font-size: 0.6rem; color: #7c9eff;
    margin-left: 0.15rem;
}

.ns-gradcam-note {
    font-size: 0.8rem; color: #3d4460; font-style: italic; margin-bottom: 1.2rem;
}
.ns-cam-label {
    font-family: 'Space Mono', monospace; font-size: 0.56rem; color: #3d4460;
    text-transform: uppercase; letter-spacing: 0.14em; text-align: center;
    margin-top: 0.5rem;
}
.ns-hr { border: none; border-top: 1px solid rgba(255,255,255,0.06); margin: 3rem 0; }

.ns-empty { text-align: center; padding: 6rem 0 4rem 0; }
.ns-empty-icon { font-size: 3.5rem; opacity: 0.08; margin-bottom: 1.2rem; }
.ns-empty-text {
    font-family: 'DM Serif Display', serif; font-size: 1rem; color: #1e2542;
}

.stSlider label {
    color: #3d4460 !important; font-size: 0.78rem !important;
    font-family: 'Space Mono', monospace !important;
}

/* Radio → toggle pill */
div[data-testid="stRadio"] { margin-bottom: 1.2rem; }
div[data-testid="stRadio"] > label { display: none; }
div[data-testid="stRadio"] > div {
    display: flex !important; gap: 0.3rem !important;
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 10px !important; padding: 0.25rem !important;
    width: fit-content !important;
}
div[data-testid="stRadio"] > div > label {
    display: block !important; font-family: 'DM Sans', sans-serif !important;
    font-size: 0.84rem !important; color: #8b91a3 !important;
    padding: 0.45rem 1.1rem !important; border-radius: 7px !important;
    cursor: pointer !important; margin: 0 !important;
}
div[data-testid="stRadio"] > div > label:has(input:checked) {
    background: #1e2542 !important; color: #f0f2f8 !important;
}
div[data-testid="stRadio"] > div > label > div:first-child { display: none !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="ns-header">
    <div class="ns-wordmark">Neuro<span class="scan">Scan</span> AI</div>
    <div class="ns-tagline">Brain Tumor Detection<br>Powered by Deep Learning</div>
</div>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_nifti(uploaded):
    import nibabel as nib
    import tempfile
    suffix = ".nii.gz" if uploaded.name.endswith(".gz") else ".nii"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name
    try:
        img = nib.load(tmp_path)
        data = img.get_fdata(dtype=np.float32)
    finally:
        os.unlink(tmp_path)
    if data.ndim == 4:
        data = data[:, :, :, 0]
    return np.rot90(data, k=1, axes=(0, 1))


def adaptive_preprocess(img: np.ndarray):
    steps = []
    out = cv2.normalize(img.copy(), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    steps.append("normalisation")

    if out.mean() < 80:
        out = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(out)
        steps.append("CLAHE (dark scan)")
    else:
        mask = out > 15
        if mask.any():
            out[mask] = cv2.equalizeHist(out[mask].reshape(-1, 1)).ravel()
        steps.append("histogram eq.")

    lap_var = cv2.Laplacian(out, cv2.CV_64F).var()
    if lap_var > 500:
        out = cv2.fastNlMeansDenoising(out, h=15, templateWindowSize=7, searchWindowSize=21)
        steps.append("NLM denoising (noisy)")
    elif lap_var > 150:
        out = cv2.GaussianBlur(out, (3, 3), 0)
        steps.append("Gaussian 3×3")
    else:
        steps.append("no denoising")

    return out, " · ".join(steps)


# ── Step 01: Upload ───────────────────────────────────────────────────────────
st.markdown('<p class="ns-eyebrow">// 01</p><p class="ns-title">Upload MRI Scan</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "upload",
    type=["jpg", "jpeg", "png", "webp", "dcm", "nii", "gz"],
    label_visibility="collapsed"
)
st.markdown('<p class="ns-upload-hint">JPG · PNG · WebP · DICOM (.dcm) · NIfTI (.nii .nii.gz)</p>',
            unsafe_allow_html=True)

if uploaded_file is None:
    st.markdown("""
    <div class="ns-empty">
        <div class="ns-empty-icon">⬡</div>
        <p class="ns-empty-text">Upload a scan above to begin</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

fname = uploaded_file.name.lower()
is_dicom = fname.endswith(".dcm")
is_nifti = fname.endswith(".nii") or fname.endswith(".nii.gz")
pixel_spacing_mm = 1.0
pixel_array = None
img_array = None

if is_dicom:
    dicom = pydicom.dcmread(uploaded_file)
    pixel_array = dicom.pixel_array.squeeze()
    try:
        pixel_spacing_mm = float(dicom.PixelSpacing[0])
    except Exception:
        pixel_spacing_mm = 1.0
elif is_nifti:
    pixel_array = load_nifti(uploaded_file)
else:
    img = Image.open(uploaded_file)
    img_array = np.array(img.convert("L")).squeeze()

# ── Multi-slice scroller ──────────────────────────────────────────────────────
if pixel_array is not None and pixel_array.ndim == 3:
    n_slices = pixel_array.shape[2] if is_nifti else pixel_array.shape[0]
    st.markdown('<div class="ns-hr"></div>', unsafe_allow_html=True)
    st.markdown(f'<p class="ns-eyebrow">Multi-Slice · {n_slices} slices detected</p>',
                unsafe_allow_html=True)

    if "slice_idx" not in st.session_state:
        st.session_state.slice_idx = n_slices // 2

    col_prev, col_slider, col_next = st.columns([1, 12, 1])
    with col_prev:
        st.write("")
        if st.button("◀"):
            st.session_state.slice_idx = max(0, st.session_state.slice_idx - 1)
    with col_slider:
        st.session_state.slice_idx = st.slider(
            "slice", 0, n_slices - 1,
            min(st.session_state.slice_idx, n_slices - 1),
            label_visibility="collapsed"
        )
    with col_next:
        st.write("")
        if st.button("▶"):
            st.session_state.slice_idx = min(n_slices - 1, st.session_state.slice_idx + 1)

    st.caption(f"Slice {st.session_state.slice_idx + 1} / {n_slices}")
    img_array = (pixel_array[:, :, st.session_state.slice_idx] if is_nifti
                 else pixel_array[st.session_state.slice_idx])
elif pixel_array is not None:
    img_array = pixel_array

img_array = cv2.normalize(img_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# ── Scan preview + info ───────────────────────────────────────────────────────
st.markdown('<div class="ns-hr"></div>', unsafe_allow_html=True)
col_img, col_info = st.columns([3, 2])
with col_img:
    st.markdown('<div class="ns-img-wrap">', unsafe_allow_html=True)
    st.image(img_array, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
with col_info:
    h_o, w_o = img_array.shape[:2]
    fmt = "DICOM" if is_dicom else "NIfTI" if is_nifti else "Image"
    brightness = round(float(img_array.mean()), 1)
    noise = round(float(cv2.Laplacian(img_array, cv2.CV_64F).var()), 1)
    st.markdown(f"""
    <div class="ns-info-card">
        <div class="ns-info-row">
            <span class="ns-info-key">File</span>
            <span class="ns-info-val">{uploaded_file.name}</span>
        </div>
        <div class="ns-info-row">
            <span class="ns-info-key">Format</span>
            <span class="ns-info-val">{fmt}</span>
        </div>
        <div class="ns-info-row">
            <span class="ns-info-key">Dimensions</span>
            <span class="ns-info-val">{w_o} × {h_o} px</span>
        </div>
        <div class="ns-info-row">
            <span class="ns-info-key">Mean Brightness</span>
            <span class="ns-info-val">{brightness}</span>
        </div>
        <div class="ns-info-row">
            <span class="ns-info-key">Noise (Laplacian)</span>
            <span class="ns-info-val">{noise}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Step 02: Preprocessing ────────────────────────────────────────────────────
st.markdown('<div class="ns-hr"></div>', unsafe_allow_html=True)
st.markdown('<p class="ns-eyebrow">// 02</p><p class="ns-title">Preprocessing</p>',
            unsafe_allow_html=True)

mode = st.radio("mode", ["Auto", "Manual"], horizontal=True,
                label_visibility="collapsed", key="preprocess_radio")

if mode == "Auto":
    denoised, steps_applied = adaptive_preprocess(img_array)
    st.markdown('<span class="ns-auto-badge">AUTO</span>', unsafe_allow_html=True)
    st.markdown(f'<p class="ns-applied"><b>Applied:</b> {steps_applied}</p>',
                unsafe_allow_html=True)
else:
    st.markdown('<div class="ns-manual-card">', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        do_norm = st.checkbox("Normalize", value=True)
        do_eq = st.checkbox("Histogram equalization")
        do_med = st.checkbox("Median filter")
    with col_b:
        do_clahe = st.checkbox("CLAHE contrast")
        do_gauss = st.checkbox("Gaussian blur")
        do_nlm = st.checkbox("Non-local means")
    st.markdown('</div>', unsafe_allow_html=True)

    proc = img_array.copy()
    if do_norm:
        proc = cv2.normalize(proc, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if do_clahe:
        proc = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(proc)
    if do_eq:
        m = proc > 15
        if m.any():
            proc[m] = cv2.equalizeHist(proc[m].reshape(-1, 1)).ravel()

    denoised = proc.copy()
    if do_gauss:
        gk = st.slider("Gaussian kernel", 1, 15, 3, step=2)
        denoised = cv2.GaussianBlur(denoised, (gk, gk), 0)
    if do_med:
        mk = st.slider("Median kernel", 1, 15, 3, step=2)
        denoised = cv2.medianBlur(denoised, mk)
    if do_nlm:
        hs = st.slider("NLM strength", 1, 30, 10)
        denoised = cv2.fastNlMeansDenoising(denoised, h=hs)

col1, col2 = st.columns(2)
with col1:
    st.markdown('<p class="ns-img-label">ORIGINAL</p>', unsafe_allow_html=True)
    st.markdown('<div class="ns-img-wrap">', unsafe_allow_html=True)
    st.image(img_array, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<p class="ns-img-label-active">PROCESSED</p>', unsafe_allow_html=True)
    st.markdown('<div class="ns-img-wrap">', unsafe_allow_html=True)
    st.image(denoised, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Step 03: Diagnosis ────────────────────────────────────────────────────────
st.markdown('<div class="ns-hr"></div>', unsafe_allow_html=True)
st.markdown('<p class="ns-eyebrow">// 03</p><p class="ns-title">Diagnosis</p>',
            unsafe_allow_html=True)

if st.button("Run Diagnosis ▶"):
    st.session_state.diagnosis_done = True

if st.session_state.get("diagnosis_done"):
    try:
        from predict import (load_model, preprocess, predict,
                             get_gradcam, overlay_gradcam)

        model = load_model("brain_tumor_detector.keras")
        input_tensor = preprocess(denoised)
        gray_128 = cv2.resize(denoised, (128, 128)).astype(np.float32) / 255.0

        with st.spinner("Analysing..."):
            class_name, confidence, all_probs = predict(model, input_tensor)

        detected = class_name.lower().replace(" ", "")

        if detected == "notumor":
            st.markdown(f"""
            <div class="ns-result-clear">
                <p class="ns-result-eyebrow ns-eb-clear">// Clear Scan</p>
                <p class="ns-result-name-clear">No Tumor Detected</p>
                <div class="ns-rdiv-clear"></div>
                <p class="ns-conf-clear">{confidence*100:.1f}% confidence</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="ns-result-tumor">
                <p class="ns-result-eyebrow ns-eb-tumor">// Tumor Detected</p>
                <p class="ns-result-name-tumor">{class_name}</p>
                <div class="ns-rdiv"></div>
                <p class="ns-conf-tumor">{confidence*100:.1f}% confidence</p>
            </div>
            """, unsafe_allow_html=True)

        # ── Step 04: Grad-CAM ─────────────────────────────────────────────────
        st.markdown('<div class="ns-hr"></div>', unsafe_allow_html=True)
        st.markdown('<p class="ns-eyebrow">// 04</p><p class="ns-title">Grad-CAM Explanation</p>',
                    unsafe_allow_html=True)
        st.markdown('<p class="ns-gradcam-note">Regions highlighted in red most influenced '
                    'the model\'s prediction.</p>', unsafe_allow_html=True)

        with st.spinner("Generating heatmap..."):
            class_idx = int(np.argmax(all_probs))
            heatmap = get_gradcam(model, input_tensor, class_idx)
            overlaid = overlay_gradcam(gray_128, heatmap)

        g1, g2, g3 = st.columns(3)
        for col, im, label in [
            (g1, gray_128, "Original Scan"),
            (g2, heatmap, "Activation Heatmap"),
            (g3, overlaid, "Overlay"),
        ]:
            with col:
                st.markdown('<div class="ns-img-wrap">', unsafe_allow_html=True)
                st.image(im, clamp=True, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown(f'<p class="ns-cam-label">{label}</p>', unsafe_allow_html=True)

        buf = io.BytesIO()
        Image.fromarray(overlaid).save(buf, format="PNG")
        st.download_button(
            "⬇  Download Grad-CAM Report",
            data=buf.getvalue(),
            file_name=f"gradcam_{uploaded_file.name.rsplit('.', 1)[0]}.png",
            mime="image/png"
        )

        # ── Step 05: Region Annotation (canvas bbox, no segmentation) ─────────
        if detected != "notumor":
            st.markdown('<div class="ns-hr"></div>', unsafe_allow_html=True)
            st.markdown('<p class="ns-eyebrow">// 05</p>'
                        '<p class="ns-title">Region Annotation</p>',
                        unsafe_allow_html=True)
            st.markdown('<p class="ns-gradcam-note">Use the Grad-CAM heatmap above as a guide. '
                        'Click and drag to draw a bounding box around the tumor region.</p>',
                        unsafe_allow_html=True)

            img_rgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            h_s, w_s = img_rgb.shape[:2]

            DISPLAY_W = 520
            display_h = int(h_s * DISPLAY_W / w_s)
            disp = Image.fromarray(img_rgb).resize((DISPLAY_W, display_h))
            _b = io.BytesIO()
            disp.save(_b, format="PNG")
            img_b64 = base64.b64encode(_b.getvalue()).decode()

            sx = round(w_s / DISPLAY_W, 6)
            sy = round(h_s / display_h, 6)

            canvas_html = f"""
            <style>
              @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500&family=Space+Mono&display=swap');
              #cv {{
                border: 1px solid rgba(124,158,255,0.18);
                border-radius: 10px; cursor: crosshair; display: block;
                background: #090c14;
              }}
              #coords {{
                font-family: 'Space Mono', monospace; font-size: 11px;
                color: #3d4460; margin-top: 8px; min-height: 16px;
                letter-spacing: 0.06em;
              }}
              #confirm {{
                margin-top: 10px; padding: 8px 20px;
                background: rgba(124,158,255,0.08); color: #7c9eff;
                border: 1px solid rgba(124,158,255,0.25);
                border-radius: 8px; cursor: pointer;
                font-family: 'DM Sans', sans-serif; font-size: 13px; font-weight: 500;
              }}
              #confirm:hover {{ background: rgba(124,158,255,0.16); }}
              #done {{
                font-family: 'Space Mono', monospace; font-size: 11px;
                color: #64dcb4; margin-left: 12px; letter-spacing: 0.06em;
              }}
            </style>
            <canvas id="cv" width="{DISPLAY_W}" height="{display_h}"></canvas>
            <div id="coords"></div>
            <button id="confirm" onclick="confirmBox()">Confirm Box</button>
            <span id="done"></span>
            <script>
            (function() {{
              const c = document.getElementById('cv');
              const ctx = c.getContext('2d');
              const im = new Image();
              let sx0, sy0, drawing = false, box = null;
              im.onload = () => ctx.drawImage(im, 0, 0);
              im.src = 'data:image/png;base64,{img_b64}';

              c.addEventListener('mousedown', e => {{
                const r = c.getBoundingClientRect();
                sx0 = e.clientX - r.left; sy0 = e.clientY - r.top;
                drawing = true;
              }});
              c.addEventListener('mousemove', e => {{
                if (!drawing) return;
                const r = c.getBoundingClientRect();
                const cx = e.clientX - r.left, cy = e.clientY - r.top;
                ctx.clearRect(0, 0, c.width, c.height);
                ctx.drawImage(im, 0, 0);
                ctx.strokeStyle = '#7c9eff';
                ctx.lineWidth = 2;
                ctx.setLineDash([5, 3]);
                ctx.strokeRect(sx0, sy0, cx - sx0, cy - sy0);
                box = {{
                  x1: Math.round(Math.min(sx0, cx)), y1: Math.round(Math.min(sy0, cy)),
                  x2: Math.round(Math.max(sx0, cx)), y2: Math.round(Math.max(sy0, cy))
                }};
                document.getElementById('coords').innerText =
                  '(' + box.x1 + ', ' + box.y1 + ') → (' + box.x2 + ', ' + box.y2 + ')';
              }});
              c.addEventListener('mouseup', () => {{ drawing = false; }});

              window.confirmBox = function() {{
                if (!box) {{ alert('Draw a box first.'); return; }}
                const x1 = Math.round(box.x1 * {sx}), y1 = Math.round(box.y1 * {sy});
                const x2 = Math.round(box.x2 * {sx}), y2 = Math.round(box.y2 * {sy});
                const val = x1 + ',' + y1 + ',' + x2 + ',' + y2;
                const inputs = window.parent.document.querySelectorAll('input[type="text"]');
                for (let inp of inputs) {{
                  if (inp.placeholder === '__bbox__') {{
                    const setter = Object.getOwnPropertyDescriptor(
                      window.HTMLInputElement.prototype, 'value').set;
                    setter.call(inp, val);
                    inp.dispatchEvent(new Event('input', {{ bubbles: true }}));
                    document.getElementById('done').innerText = '✓ CONFIRMED';
                    return;
                  }}
                }}
                document.getElementById('done').innerText = val;
              }};
            }})();
            </script>
            """
            components.html(canvas_html, height=display_h + 110)

            bbox_raw = st.text_input("bbox", placeholder="__bbox__",
                                     key="bbox_receiver", label_visibility="collapsed")

            bbox = None
            if bbox_raw and bbox_raw != "__bbox__":
                try:
                    parts = [int(p.strip()) for p in bbox_raw.split(",")]
                    if len(parts) == 4 and parts[0] < parts[2] and parts[1] < parts[3]:
                        bbox = parts
                except Exception:
                    bbox = None

            if bbox:
                x1, y1, x2, y2 = bbox
                annotated = img_rgb.copy()
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (124, 158, 255), 2)

                st.markdown('<p class="ns-img-label-active">ANNOTATED REGION</p>',
                            unsafe_allow_html=True)
                st.markdown('<div class="ns-img-wrap">', unsafe_allow_html=True)
                st.image(annotated, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

                bw = (x2 - x1) * pixel_spacing_mm
                bh = (y2 - y1) * pixel_spacing_mm
                area = bw * bh
                diag = float(np.hypot(bw, bh))
                unit = "mm" if pixel_spacing_mm != 1.0 else "px"

                st.markdown(f"""
                <div class="ns-metrics">
                    <div class="ns-metric">
                        <p class="ns-metric-key">Width</p>
                        <p class="ns-metric-val">{bw:.0f}<span class="ns-metric-unit">{unit}</span></p>
                    </div>
                    <div class="ns-metric">
                        <p class="ns-metric-key">Height</p>
                        <p class="ns-metric-val">{bh:.0f}<span class="ns-metric-unit">{unit}</span></p>
                    </div>
                    <div class="ns-metric">
                        <p class="ns-metric-key">Region Area</p>
                        <p class="ns-metric-val">{area:,.0f}<span class="ns-metric-unit">{unit}²</span></p>
                    </div>
                    <div class="ns-metric">
                        <p class="ns-metric-key">Diagonal</p>
                        <p class="ns-metric-val">{diag:.0f}<span class="ns-metric-unit">{unit}</span></p>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                buf_a = io.BytesIO()
                Image.fromarray(annotated).save(buf_a, format="PNG")
                st.markdown('<div style="margin-top:1.2rem;"></div>', unsafe_allow_html=True)
                st.download_button(
                    "⬇  Download Annotated Scan",
                    data=buf_a.getvalue(),
                    file_name=f"annotated_{uploaded_file.name.rsplit('.', 1)[0]}.png",
                    mime="image/png",
                    key="dl_annotated"
                )

    except Exception as e:
        st.error(f"Diagnosis failed: {e}")
        st.info("Make sure `brain_tumor_detector.keras` and `predict.py` are in the repo root.")
