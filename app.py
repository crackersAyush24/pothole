import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os

# Page Configuration
st.set_page_config(
    page_title="AI Pothole Detection",
    page_icon="🕳️",
    layout="wide"
)

# Header Section
st.title("🕳️ AI Road Pothole Detection System")
st.write("Upload an image of a road to automatically detect and localize potholes using your trained YOLOv8 model.")

# Sidebar Controls
st.sidebar.header("Model Settings")

# Default weight path pointing to your latest run
DEFAULT_MODEL_PATH = "/Users/ayushchaubey/Desktop/pothole/runs/detect/pothole_model-3/weights/best.pt"

model_path = st.sidebar.text_input("Model Weights Path:", value=DEFAULT_MODEL_PATH)

conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.05,
    max_value=1.00,
    value=0.25,
    step=0.05,
    help="Filter out detections below this confidence score."
)

# Load YOLO Model with caching
@st.cache_resource
def load_yolo_model(path):
    if not os.path.exists(path):
        return None
    return YOLO(path)

model = load_yolo_model(model_path)

if model is None:
    st.error(f"❌ Model file not found at `{model_path}`. Please verify the path or wait for training to complete.")
    st.stop()
else:
    st.sidebar.success("✅ Model Loaded Successfully!")

# File Uploader
uploaded_file = st.file_uploader("Choose a road image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read Image
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    # Perform Detection
    with st.spinner("Analyzing image for potholes..."):
        results = model.predict(source=img_array, conf=conf_threshold, device="mps")
        
        # Process Detections
        result = results[0]
        boxes = result.boxes
        num_potholes = len(boxes)
        
        # Render Bounding Boxes on Image
        res_plotted = result.plot()
        res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

    # Display Metrics
    col_metric1, col_metric2 = st.columns(2)
    with col_metric1:
        st.metric(label="Total Potholes Detected", value=num_potholes)
    with col_metric2:
        st.metric(label="Selected Confidence", value=f"{conf_threshold:.2f}")

    st.markdown("---")

    # Display Side-by-Side Images
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Detections Output")
        st.image(res_rgb, use_container_width=True)

    # Detailed Detection Table
    if num_potholes > 0:
        st.subheader("Detection Details")
        det_data = []
        for idx, box in enumerate(boxes):
            conf = float(box.conf[0])
            xywh = box.xywh[0].tolist()
            det_data.append({
                "Pothole #": idx + 1,
                "Confidence": f"{conf:.2%}",
                "Bounding Box (x, y, w, h)": f"{int(xywh[0])}, {int(xywh[1])}, {int(xywh[2])}, {int(xywh[3])}"
            })
        st.table(det_data)
    else:
        st.info("No potholes detected in this image above the set confidence threshold.")