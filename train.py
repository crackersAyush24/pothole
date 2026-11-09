from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")

    model.train(
        data="/Users/ayushchaubey/Desktop/pothole/data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        name="pothole_model",
        save=True,
        device="mps"
    )

if __name__ == "__main__":
    main()
