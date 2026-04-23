import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Waste System", layout="wide")

# ---------------- STYLES ----------------
st.markdown("""
<style>
html, body {
    background: linear-gradient(135deg, #0f172a, #020617);
    color: white;
    font-family: 'Inter', sans-serif;
}

.title {
    text-align:center;
    font-size:46px;
    font-weight:700;
    background: linear-gradient(90deg, #22c55e, #06b6d4);
    -webkit-background-clip: text;
    color: transparent;
}

.subtitle {
    text-align:center;
    color:#9ca3af;
    margin-bottom:25px;
}

.glass {
    background: rgba(255,255,255,0.05);
    border-radius: 18px;
    padding: 18px;
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 18px;
}

/* FILE UPLOADER */
[data-testid="stFileUploader"] {
    border: 2px dashed rgba(255,255,255,0.25);
    padding: 45px;
    border-radius: 20px;
    text-align: center;
    transition: all 0.3s ease;
}

/* HOVER */
[data-testid="stFileUploader"]:hover {
    border-color: #22c55e;
    box-shadow: 0px 0px 25px rgba(34,197,94,0.3);
    transform: scale(1.02);
}

/* GLOW ANIMATION */
@keyframes glowPulse {
    0% { box-shadow: 0 0 10px rgba(34,197,94,0.1); }
    50% { box-shadow: 0 0 25px rgba(34,197,94,0.35); }
    100% { box-shadow: 0 0 10px rgba(34,197,94,0.1); }
}

[data-testid="stFileUploader"] {
    animation: glowPulse 2.5s infinite;
}

/* PROGRESS BAR */
.stProgress > div > div > div {
    background-image: linear-gradient(90deg, #22c55e, #06b6d4);
}
</style>
""", unsafe_allow_html=True)

# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("waste_model.h5", compile=False)

model = load_model()

# ---------------- CLASSES ----------------
class_names = ['cardboard', 'glass', 'metal', 'organic waste', 'paper', 'plastic', 'trash']

disposal_map = {
    'cardboard': '♻️ Recycle Bin',
    'glass': '♻️ Recycle Bin',
    'metal': '♻️ Recycle Bin',
    'paper': '♻️ Recycle Bin',
    'plastic': '♻️ Recycle Bin',
    'trash': '🗑️ Landfill',
    'organic waste': '🌱 Compost'
}

# ---------------- SESSION ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- HEADER ----------------
st.markdown("<div class='title'>♻️ AI Waste Intelligence</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Smart • Explainable • Real-world Waste Classification</div>", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.5, 0.9, 0.6)
gap_threshold = st.sidebar.slider("Uncertainty Gap", 0.05, 0.3, 0.15)

# ---------------- INPUT ----------------
st.markdown("### 📤 Upload Image")
uploaded_file = st.file_uploader("Drag & drop or click to upload", type=["jpg","png","jpeg"])

st.markdown("### 📷 Camera")
use_camera = st.toggle("Enable Camera")

camera_image = None
if use_camera:
    camera_image = st.camera_input("Capture Image")

input_image = uploaded_file if uploaded_file else camera_image

# ---------------- MAIN ----------------
if input_image:

    col_left, col_right = st.columns([1,1])

    with col_left:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.image(input_image, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)

        img = Image.open(input_image).convert("RGB")
        img = img.resize((224,224))
        img_array = np.array(img)/255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)[0]

        top_indices = np.argsort(predictions)[::-1]
        top1_idx, top2_idx = top_indices[0], top_indices[1]

        top1_conf = predictions[top1_idx]
        top2_conf = predictions[top2_idx]

        predicted_class = class_names[top1_idx]

        if top1_conf < confidence_threshold:
            final_label = "unknown"
        elif (top1_conf - top2_conf) < gap_threshold:
            final_label = "uncertain"
        else:
            final_label = predicted_class

        if final_label == "unknown":
            organic_conf = predictions[class_names.index("organic waste")]
            if organic_conf > 0.25:
                final_label = "organic waste"

        gap = top1_conf - top2_conf
        reliability = "High" if gap > 0.3 else "Medium" if gap > 0.15 else "Low"

        st.markdown("### 🧠 Final Decision")

        if final_label == "unknown":
            st.error("Unknown Waste")
        elif final_label == "uncertain":
            st.warning(f"Uncertain → {predicted_class}")
        else:
            st.success(final_label.upper())

        st.write(f"Confidence: {top1_conf*100:.2f}%")
        st.write(f"Reliability: {reliability}")

        if final_label in disposal_map:
            st.markdown("### ♻️ Disposal")
            st.success(disposal_map[final_label])

        st.markdown("</div>", unsafe_allow_html=True)

    # CONFIDENCE
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.markdown("### 🔍 Confidence Breakdown")

    for i in top_indices[:3]:
        st.markdown(f"{class_names[i]} — {predictions[i]*100:.2f}%")
        st.progress(int(predictions[i]*100))

    st.markdown("</div>", unsafe_allow_html=True)

    # SAVE HISTORY
    st.session_state.history.append((final_label, top1_conf))

# ---------------- HISTORY + ANALYTICS (BOTTOM) ----------------
if "history" in st.session_state and st.session_state.history:

    st.markdown("## 📊 Analytics")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)

        col_h1, col_h2 = st.columns([3,1])

        with col_h1:
            st.markdown("### Prediction History")

        with col_h2:
            if st.button("🧹 Clear"):
                st.session_state.history = []
                st.rerun()

        for i,(cls,conf) in enumerate(reversed(st.session_state.history[-5:])):
            st.write(f"{i+1}. {cls} ({conf*100:.2f}%)")

        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown("### Waste Distribution")

        labels = [x[0] for x in st.session_state.history]
        counts = {l: labels.count(l) for l in set(labels)}

        fig, ax = plt.subplots()
        ax.pie(counts.values(), labels=counts.keys(), autopct='%1.1f%%')
        ax.axis('equal')

        st.pyplot(fig)

        st.markdown("</div>", unsafe_allow_html=True)


# ---------------- DOWNLOAD REPORT ----------------
if "history" in st.session_state and st.session_state.history:

    st.markdown("### 📥 Download Report")

    import pandas as pd

    data = [
        {
            "Prediction": cls,
            "Confidence (%)": round(conf * 100, 2)
        }
        for cls, conf in st.session_state.history
    ]

    df = pd.DataFrame(data)

    csv = df.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="⬇️ Download CSV Report",
        data=csv,
        file_name="waste_predictions_report.csv",
        mime="text/csv"
    )

# ---------------- FOOTER ----------------
st.markdown("<center style='color:#888;'>Final AI Waste System 🚀</center>", unsafe_allow_html=True)