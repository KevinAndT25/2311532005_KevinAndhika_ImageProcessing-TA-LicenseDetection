import streamlit as st
import numpy as np

from PIL import Image
from utils.detector import predict_image

st.title("🔍 License Plate Detection")

tab1, tab2 = st.tabs([
    "📁 Upload Image",
    "📷 Camera"
])

# ==================================
# UPLOAD
# ==================================

with tab1:

    uploaded_file = st.file_uploader(
        "Upload Gambar",
        type=["jpg","jpeg","png"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file)

        image_np = np.array(image)

        result = predict_image(image_np)

        st.subheader("Hasil Deteksi")

        col1,col2 = st.columns(2)

        with col1:

            st.image(
                result["original"],
                caption="Gambar Asli",
                use_container_width=True
            )

        with col2:

            st.image(
                result["detected"],
                caption="Hasil Deteksi",
                use_container_width=True
            )

        st.divider()

        m1,m2,m3,m4 = st.columns(4)

        m1.metric(
            "Jumlah Deteksi",
            result["detections"]
        )

        m2.metric(
            "Confidence Score",
            f"{result['confidence']:.2%}"
        )

        m3.metric(
            "Waktu Inferensi",
            f"{result['inference_time']:.3f}s"
        )

        m4.metric(
            "Model",
            "YOLOv8"
        )

# ==================================
# CAMERA
# ==================================

with tab2:

    camera = st.camera_input(
        "Ambil Foto"
    )

    if camera:

        image = Image.open(camera)

        image_np = np.array(image)

        result = predict_image(image_np)

        col1,col2 = st.columns(2)

        with col1:

            st.image(
                result["original"],
                caption="Gambar Asli",
                use_container_width=True
            )

        with col2:

            st.image(
                result["detected"],
                caption="Hasil Deteksi",
                use_container_width=True
            )

        st.divider()

        m1,m2,m3,m4 = st.columns(4)

        m1.metric(
            "Jumlah Terdeteksi",
            result["detections"]
        )

        m2.metric(
            "Confidence Score",
            f"{result['confidence']:.2%}"
        )

        m3.metric(
            "Waktu Inferensi",
            f"{result['inference_time']:.3f}s"
        )

        m4.metric(
            "Model",
            "YOLOv8"
        )