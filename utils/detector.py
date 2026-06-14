from ultralytics import YOLO
import time
import numpy as np

MODEL_PATH = "best.pt"

model = YOLO(MODEL_PATH)

def predict_image(image):

    start_time = time.time()

    results = model.predict(
        image,
        conf=0.25,
        verbose=False
    )

    end_time = time.time()

    inference_time = end_time - start_time

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