from ultralytics import YOLO
import streamlit as st
import time

MODEL_PATH = "best.pt"

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

def predict_image(image):

    start_time = time.time()

    results = model.predict(
        image,
        conf=0.25,
        verbose=False
    )

    inference_time = time.time() - start_time

    result = results[0]

    plotted_image = result.plot()

    detections = len(result.boxes)

    confidence = 0

    if detections > 0:
        confidence = float(
            result.boxes.conf.max().cpu().numpy()
        )

    return {
        "original": image,
        "detected": plotted_image,
        "confidence": confidence,
        "detections": detections,
        "inference_time": inference_time
    }

# import cv2

# def predict_image(image):
#     return {
#         "original": image,
#         "detected": image,
#         "confidence": 1.0,
#         "detections": 0,
#         "inference_time": 0
#     }