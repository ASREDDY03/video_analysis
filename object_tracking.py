import os
import cv2
import pandas as pd
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLOv8 model and DeepSort tracker
model = YOLO("yolov8n.pt")
tracker = DeepSort(max_age=30)

# Output directory and CSV path
DEBUG_OUTPUTS_DIR = "debug_outputs"
os.makedirs(DEBUG_OUTPUTS_DIR, exist_ok=True)
TRACKING_CSV_PATH = os.path.join(DEBUG_OUTPUTS_DIR, "object_tracking_log.csv")

def run_object_tracking(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_id = 0
    log_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        time_sec = round(frame_id / fps, 2)
        try:
            results = model(frame, verbose=False)[0]
        except Exception as e:
            print(f"Skipping frame {frame_id} due to YOLO error: {e}")
            frame_id += 1
            continue

        detections = []
        for box in results.boxes:
            try:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = model.names[cls]
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, label))
            except Exception as e:
                print(f"Detection error at frame {frame_id}: {e}")
                continue

        try:
            tracks = tracker.update_tracks(detections, frame=frame)
        except Exception as e:
            print(f"Tracking error at frame {frame_id}: {e}")
            frame_id += 1
            continue

        for track in tracks:
            if not track.is_confirmed():
                continue

            try:
                track_id = track.track_id
                l, t, r, b = track.to_ltrb()
                label = track.get_det_class()
                box_width = r - l
                box_height = b - t
                center_x = l + box_width / 2
                center_y = t + box_height / 2

                log_data.append({
                    "frame": frame_id,
                    "time_sec": time_sec,
                    "track_id": track_id,
                    "class": label,
                    "x1": int(l), "y1": int(t), "x2": int(r), "y2": int(b),
                    "center_x": int(center_x), "center_y": int(center_y),
                    "box_width": int(box_width), "box_height": int(box_height)
                })
            except Exception as e:
                print(f"Error logging track at frame {frame_id}: {e}")
                continue

        frame_id += 1

    cap.release()

    # Save tracking logs to CSV
    df = pd.DataFrame(log_data)
    df.to_csv(TRACKING_CSV_PATH, index=False)
    print(f"Object tracking saved to {TRACKING_CSV_PATH}")

if __name__ == "__main__":
    test_video = "uploads/uploaded_video.mp4"
    run_object_tracking(test_video)
