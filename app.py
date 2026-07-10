import streamlit as st
import numpy as np
from PIL import Image
import cv2
import io
import pydicom

st.set_page_config(
    page_title="NeuroScan AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500&family=DM+Serif+Display&family=Space+Mono&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0C0F1A;
    color: #C9CDD9;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 3rem 4rem 3rem; max-width: 1080px; }

/* ── Header ───────────────────────────────────────────── */
.ns-header {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    padding: 2.8rem 0 2rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 3rem;
}
.ns-wordmark {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    color: #F0F2F8;
    letter-spacing: -0.01em;
    line-height: 1;
}
.ns-wordmark em {
    font-style: normal;
    color: #7C9EFF;
}
.ns-tagline {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: #3D4460;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    line-height: 1.6;
    text-align: right;
}

/* ── Step headers ─────────────────────────────────────── */
.ns-step-num {
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    color: #7C9EFF;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-bottom: 0.25rem;
}
.ns-step-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.45rem;
    color: #F0F2F8;
    margin-bottom: 1.1rem;
    font-weight: 400;
    letter-spacing: -0.01em;
}

/* ── Upload area ──────────────────────────────────────── */
.ns-upload-hint {
    font-size: 0.75rem;
    color: #3D4460;
    margin-top: 0.4rem;
    letter-spacing: 0.03em;
}

/* ── Info card ────────────────────────────────────────── */
.ns-info-card {
    background: linear-gradient(160deg, #131829 0%, #0E1220 100%);
    border: 1px solid rgba(124,158,255,0.12);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
}
.ns-info-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    padding: 0.35rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
}
.ns-info-row:last-child { border-bottom: none; }
.ns-info-key {
    font-size: 0.72rem;
    color: #3D4460;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-family: 'Space Mono', monospace;
}
.ns-info-val {
    font-size: 0.82rem;
    color: #C9CDD9;
    text-align: right;
    max-width: 60%;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

/* ── Preprocessing badge ──────────────────────────────── */
.ns-auto-tag {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(124,158,255,0.08);
    border: 1px solid rgba(124,158,255,0.2);
    border-radius: 20px;
    padding: 0.25rem 0.75rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem;
    color: #7C9EFF;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.ns-applied {
    font-size: 0.76rem;
    color: #3D4460;
    margin-top: 0.2rem;
    font-style: italic;
}

/* ── Result cards ─────────────────────────────────────── */
.ns-result-tumor {
    position: relative;
    overflow: hidden;
    background: linear-gradient(135deg, #131829 0%, #160E28 100%);
    border: 1px solid rgba(180,120,255,0.3);
    border-radius: 16px;
    padding: 2.5rem 2rem;
    text-align: center;
    margin: 1.2rem 0;
}
.ns-result-tumor::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(140,80,255,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.ns-result-clear {
    position: relative;
    overflow: hidden;
    background: linear-gradient(135deg, #0D1A1A 0%, #0A1520 100%);
    border: 1px solid rgba(100,220,180,0.25);
    border-radius: 16px;
    padding: 2.5rem 2rem;
    text-align: center;
    margin: 1.2rem 0;
}
.ns-result-clear::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(100,220,180,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.ns-result-eyebrow {
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}
.ns-result-eyebrow-tumor { color: #9B6BFF; }
.ns-result-eyebrow-clear { color: #64DCB4; }
.ns-result-name-tumor {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    color: #F0F2F8;
    line-height: 1;
    margin-bottom: 0.6rem;
    letter-spacing: -0.02em;
}
.ns-result-name-clear {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    color: #64DCB4;
    line-height: 1;
    margin-bottom: 0.6rem;
    letter-spacing: -0.02em;
}
.ns-conf-tumor {
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    color: #9B6BFF;
    letter-spacing: 0.06em;
}
.ns-conf-clear {
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    color: #64DCB4;
    letter-spacing: 0.06em;
}

/* ── Gradcam caption ──────────────────────────────────── */
.ns-gradcam-note {
    font-size: 0.78rem;
    color: #3D4460;
    margin-bottom: 1rem;
    font-style: italic;
}

/* ── Divider ──────────────────────────────────────────── */
.ns-divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.05);
    margin: 2.5rem 0;
}

/* ── Empty state ──────────────────────────────────────── */
.ns-empty {
    text-align: center;
    padding: 5rem 0 3rem 0;
}
.ns-empty-glyph {
    font-size: 4rem;
    opacity: 0.15;
    margin-bottom: 1rem;
}
.ns-empty-text {
    font-family: 'DM Serif Display', serif;
    font-size: 1.1rem;
    color: #252A3A;
    letter-spacing: 0.01em;
}

/* ── Streamlit overrides ──────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #7C9EFF 0%, #9B6BFF 100%) !important;
    color: #0C0F1A !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.65rem 2.2rem !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.03em !important;
}
.stButton > button:hover {
    opacity: 0.88 !important;
}
.stToggle label { color: #6B7280 !important; font-size: 0.84rem !important; }
.stCheckbox label { color: #6B7280 !important; font-size: 0.82rem !important; }
.stSlider label { color: #6B7280 !important; font-size: 0.82rem !important; }
[data-testid="stFileUploader"] section {
    background: linear-gradient(160deg, #131829 0%, #0E1220 100%) !important;
    border: 1.5px dashed rgba(124,158,255,0.2) !important;
    border-radius: 12px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="ns-header">
    <div class="ns-wordmark">Neuro<em>Scan</em> <span style="color:#3D4460;font-size:1rem;">AI</span></div>
    <div class="ns-tagline">Brain Tumor Detection<br>Powered by Deep Learning</div>
</div>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_nifti(uploaded):
    import nibabel as nib
    import tempfile, os
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
    data = np.rot90(data, k=1, axes=(0, 1))
    return data


def adaptive_preprocess(img: np.ndarray):
    steps = []
    out = cv2.normalize(img.copy(), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    steps.append("normalisation")

    mean_b = out.mean()
    if mean_b < 80:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        out = clahe.apply(out)
        steps.append("CLAHE (dark scan)")
    else:
        mask = out > 15
        if mask.any():
            brain_px = out[mask]
            out[mask] = cv2.equalizeHist(brain_px.reshape(-1, 1)).ravel()
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
st.markdown('<p class="ns-step-num">// 01</p><p class="ns-step-title">Upload MRI Scan</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "upload",
    type=["jpg", "jpeg", "png", "webp", "dcm", "nii", "nii.gz"],
    label_visibility="collapsed"
)
st.markdown('<p class="ns-upload-hint">JPG · PNG · WebP · DICOM (.dcm) · NIfTI (.nii .nii.gz)</p>', unsafe_allow_html=True)

if uploaded_file is not None:
    fname = uploaded_file.name.lower()
    is_dicom = fname.endswith(".dcm")
    is_nifti = fname.endswith(".nii") or fname.endswith(".nii.gz")
    pixel_array = None
    img_array = None

    if is_dicom:
        dicom = pydicom.dcmread(uploaded_file)
        pixel_array = dicom.pixel_array.squeeze()
    elif is_nifti:
        pixel_array = load_nifti(uploaded_file)
    else:
        img = Image.open(uploaded_file)
        img_array = np.array(img.convert("L")).squeeze()

    # Multi-slice scroller
    if pixel_array is not None and len(pixel_array.shape) == 3:
        n_slices = pixel_array.shape[2] if is_nifti else pixel_array.shape[0]
        st.markdown('<div class="ns-divider"></div>', unsafe_allow_html=True)
        st.markdown(f'<p class="ns-step-num">Multi-Slice · {n_slices} slices detected</p>', unsafe_allow_html=True)

        if "slice_idx" not in st.session_state:
            st.session_state.slice_idx = n_slices // 2

        col_prev, col_slider, col_next = st.columns([1, 10, 1])
        with col_prev:
            st.write("")
            if st.button("◀"):
                st.session_state.slice_idx = max(0, st.session_state.slice_idx - 1)
        with col_slider:
            st.session_state.slice_idx = st.slider(
                "slice", 0, n_slices - 1,
                st.session_state.slice_idx,
                label_visibility="collapsed"
            )
        with col_next:
            st.write("")
            if st.button("▶"):
                st.session_state.slice_idx = min(n_slices - 1, st.session_state.slice_idx + 1)

        st.caption(f"Slice {st.session_state.slice_idx + 1} / {n_slices}")
        img_array = pixel_array[:, :, st.session_state.slice_idx] if is_nifti else pixel_array[st.session_state.slice_idx]

    elif pixel_array is not None:
        img_array = pixel_array

    img_array = cv2.normalize(img_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Scan preview + info
    st.markdown('<div class="ns-divider"></div>', unsafe_allow_html=True)
    col_img, col_info = st.columns([3, 2])
    with col_img:
        st.image(img_array, use_container_width=True)
    with col_info:
        h, w = img_array.shape[:2]
        fmt = "DICOM" if is_dicom else "NIfTI" if is_nifti else "Image"
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
                <span class="ns-info-val">{w} × {h} px</span>
            </div>
            <div class="ns-info-row">
                <span class="ns-info-key">Mean brightness</span>
                <span class="ns-info-val">{img_array.mean():.1f}</span>
            </div>
            <div class="ns-info-row">
                <span class="ns-info-key">Noise (Laplacian)</span>
                <span class="ns-info-val">{cv2.Laplacian(img_array, cv2.CV_64F).var():.1f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Step 02: Preprocessing ────────────────────────────────────────────────
    st.markdown('<div class="ns-divider"></div>', unsafe_allow_html=True)
    st.markdown('<p class="ns-step-num">// 02</p><p class="ns-step-title">Preprocessing</p>', unsafe_allow_html=True)

    auto_mode = st.toggle("Auto preprocessing (recommended)", value=True)

    if auto_mode:
        denoised, steps_applied = adaptive_preprocess(img_array)
        st.markdown(f'<span class="ns-auto-tag">● Auto</span>', unsafe_allow_html=True)
        st.markdown(f'<p class="ns-applied">{steps_applied}</p>', unsafe_allow_html=True)
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            do_norm = st.checkbox("Normalize", value=True)
            do_clahe = st.checkbox("CLAHE contrast")
            do_eq = st.checkbox("Histogram equalization")
        with col_b:
            do_gauss = st.checkbox("Gaussian blur")
            do_med = st.checkbox("Median filter")
            do_nlm = st.checkbox("Non-local means")

        proc = img_array.copy()
        if do_norm:
            proc = cv2.normalize(proc, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if do_clahe:
            proc = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(proc)
        if do_eq:
            mask = proc > 15
            if mask.any():
                proc[mask] = cv2.equalizeHist(proc[mask].reshape(-1, 1)).ravel()

        denoised = proc.copy()
        if do_gauss:
            gk = st.slider("Kernel size", 1, 15, 3, step=2)
            denoised = cv2.GaussianBlur(denoised, (gk, gk), 0)
        if do_med:
            mk = st.slider("Median kernel", 1, 15, 3, step=2)
            denoised = cv2.medianBlur(denoised, mk)
        if do_nlm:
            h_s = st.slider("NLM strength", 1, 30, 10)
            denoised = cv2.fastNlMeansDenoising(denoised, h=h_s)

    col1, col2 = st.columns(2)
    with col1:
        st.image(img_array, caption="Original", use_container_width=True)
    with col2:
        st.image(denoised, caption="Processed", use_container_width=True)

    # ── Step 03: Diagnosis ────────────────────────────────────────────────────
    st.markdown('<div class="ns-divider"></div>', unsafe_allow_html=True)
    st.markdown('<p class="ns-step-num">// 03</p><p class="ns-step-title">Diagnosis</p>', unsafe_allow_html=True)

    run_btn = st.button("Run Diagnosis →")

    if run_btn:
        try:
            from predict import load_model, preprocess, predict, get_gradcam, overlay_gradcam, CLASS_NAMES
            model = load_model("brain_tumor_detector.keras")

            input_tensor = preprocess(denoised)
            gray_128 = cv2.resize(denoised, (128, 128)).astype(np.float32) / 255.0

            with st.spinner("Analysing..."):
                class_name, confidence, all_probs = predict(model, input_tensor)

            detected_class = class_name.lower().replace(" ", "")

            if detected_class == "notumor":
                st.markdown(f"""
                <div class="ns-result-clear">
                    <p class="ns-result-eyebrow ns-result-eyebrow-clear">// Diagnosis</p>
                    <p class="ns-result-name-clear">No Tumor Detected</p>
                    <p class="ns-conf-clear">{confidence*100:.1f}% confidence</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="ns-result-tumor">
                    <p class="ns-result-eyebrow ns-result-eyebrow-tumor">// Tumor Detected</p>
                    <p class="ns-result-name-tumor">{class_name}</p>
                    <p class="ns-conf-tumor">{confidence*100:.1f}% confidence</p>
                </div>
                """, unsafe_allow_html=True)

            # ── Step 04: Grad-CAM ─────────────────────────────────────────────
            st.markdown('<div class="ns-divider"></div>', unsafe_allow_html=True)
            st.markdown('<p class="ns-step-num">// 04</p><p class="ns-step-title">Grad-CAM Explanation</p>', unsafe_allow_html=True)
            st.markdown('<p class="ns-gradcam-note">Regions highlighted in red most influenced the model\'s prediction.</p>', unsafe_allow_html=True)

            with st.spinner("Generating heatmap..."):
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

            st.markdown('<div class="ns-divider"></div>', unsafe_allow_html=True)
            buf = io.BytesIO()
            Image.fromarray(overlaid).save(buf, format="PNG")
            st.download_button(
                "Download Grad-CAM Report",
                data=buf.getvalue(),
                file_name=f"gradcam_{uploaded_file.name.rsplit('.', 1)[0]}.png",
                mime="image/png"
            )

        except Exception as e:
            st.error(f"Diagnosis failed: {e}")
            st.info("Make sure `brain_tumor_detector.keras` and `predict.py` are in the repo root.")

else:
    st.markdown("""
    <div class="ns-empty">
        <div class="ns-empty-glyph">⬡</div>
        <p class="ns-empty-text">Upload a scan above to begin</p>
    </div>
    """, unsafe_allow_html=True)
