from ultralytics import YOLO

# ✅ Load YOLO model
yolo_model = YOLO("yolov8n.pt")

def detect_objects(frame):
    """Detects objects in a given video frame using YOLOv8."""
    results = yolo_model(frame)

    # ✅ Extract detected object class labels
    detected_objects = []
    for result in results:
        if hasattr(result, "names") and hasattr(result, "boxes"):
            for box in result.boxes.data.tolist():  # ✅ Extract bounding box data
                class_id = int(box[5])  # ✅ Get class ID
                detected_objects.append(result.names[class_id])  # ✅ Get object label

    return detected_objects  # ✅ Returns a clean list of object names (strings)
