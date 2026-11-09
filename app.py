import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile
import os

st.set_page_config(page_title="Pothole Detector AI", page_icon="🕳️", layout="wide")

st.title("🕳️ AI Pothole Detection System")
st.write("Detect potholes in real-time across images, video files, or a live webcam feed.")

# Sidebar Settings
st.sidebar.header("Model Configuration")
DEFAULT_MODEL_PATH = "/Users/ayushchaubey/Desktop/pothole/runs/detect/pothole_model-3/weights/best.pt"
model_path = st.sidebar.text_input("Weights Path:", value=DEFAULT_MODEL_PATH)

conf_thresh = st.sidebar.slider(
    "Confidence Threshold", 0.05, 1.00, 0.25, 0.05
)

# Load Model
@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        return None
    return YOLO(path)

model = load_model(model_path)

if model is None:
    st.error(f"❌ Model weight file not found at `{model_path}`. Please verify your path.")
    st.stop()
else:
    st.sidebar.success("✅ Model Ready")

# Tabs Navigation
tab_img, tab_vid, tab_webcam = st.tabs(["🖼️ Image Upload", "🎥 Video File", "📹 Live Webcam"])

# ==================== TAB 1: IMAGE UPLOAD ====================
with tab_img:
    st.header("Single Image Detection")
    img_file = st.file_uploader("Choose a road image", type=["jpg", "jpeg", "png"], key="img_uploader")
    
    if img_file is not None:
        image = Image.open(img_file).convert("RGB")
        img_np = np.array(image)
        
        with st.spinner("Analyzing image..."):
            results = model.predict(source=img_np, conf=conf_thresh, device="mps")
            res_plotted = results[0].plot()
            res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            num_potholes = len(results[0].boxes)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)
        with col2:
            st.subheader(f"Detections ({num_potholes} Found)")
            st.image(res_rgb, use_container_width=True)

# ==================== TAB 2: VIDEO FILE UPLOAD ====================
with tab_vid:
    st.header("Video File Detection")
    vid_file = st.file_uploader("Choose a road video", type=["mp4", "mov", "avi"], key="vid_uploader")
    
    if vid_file is not None:
        # Save video to temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(vid_file.read())
        tfile.close()

        cap = cv2.VideoCapture(tfile.name)
        st_frame = st.empty()
        
        stop_btn = st.button("Stop Video Processing", key="stop_vid")
        
        while cap.isOpened() and not stop_btn:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Predict frame
            results = model.predict(source=frame, conf=conf_thresh, device="mps", verbose=False)
            res_plotted = results[0].plot()
            res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            
            st_frame.image(res_rgb, caption="Processing Video Stream...", use_container_width=True)

        cap.release()
        os.remove(tfile.name)
        st.success("Video processing complete.")

# ==================== TAB 3: LIVE WEBCAM ====================
# ==================== TAB 3: LIVE CAMERA FEED ====================
with tab_webcam:
    st.header("Real-Time Camera Feed")
    
    cam_type = st.radio(
        "Select Camera Input Method:",
        ["Mac / iPhone Continuity Camera (Index)", "IP Camera (Wi-Fi URL Stream)"],
        horizontal=True
    )

    if cam_type == "Mac / iPhone Continuity Camera (Index)":
        cam_index = st.number_input(
            "Camera Index (0 = Built-in Mac Cam, 1 or 2 = iPhone / External Cam):",
            min_value=0,
            max_value=5,
            value=0,
            step=1
        )
        source = int(cam_index)
    else:
        source = st.text_input(
            "Enter IP Camera Stream URL (from phone app):",
            value="http://192.168.1.50:8080/video"
        )

    run_webcam = st.toggle("Start Live Feed", value=False)
    
    if run_webcam:
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            st.error(f"Could not open video stream from source: `{source}`. Verify connection or IP URL.")
        else:
            st_webcam_frame = st.empty()
            
            while run_webcam:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Failed to grab video frame. Check connection.")
                    break
                
                # Perform real-time YOLO detection
                results = model.predict(source=frame, conf=conf_thresh, device="mps", verbose=False)
                res_plotted = results[0].plot()
                res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                
                st_webcam_frame.image(res_rgb, caption="Live Detection Stream", use_container_width=True)
                
            cap.release()

