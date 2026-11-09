# 🕳️ Real-Time Pothole Detection & Localization System (YOLOv8 + Streamlit)

An end-to-end Computer Vision application built with **YOLOv8** and **Streamlit** to detect, localize, and quantify road potholes in real time. Optimized for Apple Silicon GPU acceleration (`mps`), NVIDIA CUDA (`cuda`), and standard CPUs.

---

## 📌 Features

- 🖼️ **Image Detection:** Upload single or multiple road images to identify potholes with confidence scoring and detailed bounding box coordinate metrics.
- 🎥 **Video Analytics:** Process recorded driving footage (`.mp4`, `.avi`, `.mov`) frame-by-frame with low latency overlay.
- 📹 **Live Camera & Mobile Stream:** Connect built-in webcams, **iPhone Continuity Camera**, or **Wi-Fi IP Camera** streams (Android/iOS) for live detection.
- ⚡ **Apple Silicon Accelerated:** Fully optimized to leverage Apple Neural Engine / GPU via PyTorch MPS backend.
- 📊 **Metric Dashboard:** Displays total pothole count, real-time confidence threshold adjustments, and bounding box arrays.

---

## 📁 Project Directory Structure

```text
pothole_project/
├── raw_data/                 # Downloaded Kaggle dataset (extracted images & masks)
├── dataset/                  # Processed YOLO dataset (generated automatically)
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   └── val/
│       ├── images/
│       └── labels/
├── data.yaml                 # YOLO dataset configuration file
├── prepare_data.py           # Preprocessing script: Mask-to-YOLO bounding box converter
├── train.py                  # Model training script
├── app.py                    # Multi-tab Streamlit web application
├── view_plots.py             # Evaluation plots visualization script
└── README.md                 # Project documentation


🛠️ Prerequisites & Installation

1. Requirements

⚬ Python: 3.10 or higher (Conda/Anaconda recommended)
⚬ OS: macOS (Apple Silicon M1/M2/M3/M4 or Intel), Linux, or Windows 10/11
⚬ Hardware: Apple Silicon (mps), NVIDIA GPU (cuda), or standard CPU (cpu)

2. Environment Setup

Clone this repository and install all required Python packages:

# Clone the repository
git clone [https://github.com/crackersAyush24/pothole.git)
cd pothole-detection-yolov8

# Install dependencies
pip install ultralytics opencv-python pillow pyyaml scikit-learn streamlit kaggle matplotlib torch torchvision


🚀 Quick Start Guide

Step 1: Prepare the Dataset

Download the Pothole Image Segmentation Dataset into the raw_data/ folder and run the preprocessing script to generate normalized YOLO labels and an 80/20 train/validation split:

python prepare_data.py


Step 2: Train the YOLOv8 Model

Train the model on your dataset:

python train.py


Trained weights will automatically be saved to: runs/detect/pothole_model/weights/best.pt

Step 3: Launch the Streamlit Web Application

Start the interactive dashboard:

streamlit run app.py


The application will open in your browser at http://localhost:8501.

📹 Mobile & External Camera Setup

Option A: iPhone Continuity Camera (macOS)

1. Ensure Wi-Fi and Bluetooth are enabled on both your iPhone and Mac.
2. Open the Live Camera tab in the Streamlit app.
3. Select Mac / iPhone Continuity Camera and set the index to 1 or 2.

Option B: Wi-Fi IP Camera App (Android / iOS)

1. Install an IP camera streaming app (IP Webcam for Android or IP Camera Lite for iOS).
2. Connect your phone to the same Wi-Fi network as your computer and tap Start Server.
3. Copy the stream URL provided (e.g., http://192.168.1.50:8080/video) into the IP Camera input field in Streamlit.

📈 Evaluation & Performance Metrics

Metric	Target / Typical Score	Description
Precision (P)	\ge 0.85	Ratio of true detected potholes to total predicted potholes.
Recall (R)	\ge 0.85	Ratio of actual potholes correctly detected by the model.
mAP50	\ge 0.80	Mean Average Precision at 50% Intersection-over-Union (IoU).
mAP50-95	\ge 0.50	Overall precision averaged across IoU thresholds from 50% to 95%.

To view saved training metrics, loss curves, and confusion matrices:

python view_plots.py


📜 License

This project is under ayush chaubey. Feel free to modify and distribute.
