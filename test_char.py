import os
import cv2
import numpy as np
import pandas as pd
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity

# Paths
VIDEO_PATH = "uploads/uploaded_video.mp4"
OUTPUT_LOG = "debug_outputs/test_character_log.csv"
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
FACE_DEBUG_DIR = "debug_faces"
MIN_FACE_SIZE = 50  # Skip faces smaller than 50x50 pixels
SIMILARITY_THRESHOLD = 0.4

# Prepare environment
os.makedirs("debug_outputs", exist_ok=True)
os.makedirs(FACE_DEBUG_DIR, exist_ok=True)

# Load OpenCV face detector
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

# Globals
known_characters = []
character_id_counter = 1
log_data = []

def recognize_characters(video_path):
    global character_id_counter

    if not os.path.exists(video_path):
        print("❌ Video not found at:", video_path)
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Cannot open video.")
        return

    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"\n🎥 Processing video at {fps:.2f} FPS...\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_sec = round(frame_count / fps, 2)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for i, (x, y, w, h) in enumerate(faces):
            if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
                continue  # Skip very small face regions

            face_crop = frame[y:y+h, x:x+w]

            # Optional: save face crop for debugging
            face_debug_path = f"{FACE_DEBUG_DIR}/frame_{frame_count}_face_{i}.jpg"
            cv2.imwrite(face_debug_path, face_crop)

            try:
                embedding = DeepFace.represent(
                    face_crop,
                    model_name="Facenet",
                    enforce_detection=False
                )[0]["embedding"]
            except Exception as e:
                print(f"[!] Embedding error on frame {frame_count}, face {i}: {e}")
                continue

            matched = False
            matched_id = None
            similarity_score = 0.0

            for known in known_characters:
                similarity = cosine_similarity([embedding], [known["embedding"]])[0][0]
                if similarity >= SIMILARITY_THRESHOLD:
                    matched = True
                    matched_id = known["id"]
                    similarity_score = similarity
                    break

            if not matched:
                matched_id = f"Person_{character_id_counter}"
                known_characters.append({"id": matched_id, "embedding": embedding})
                character_id_counter += 1

            log_data.append({
                "frame": frame_count,
                "time_sec": timestamp_sec,
                "face_index": i,
                "character_id": matched_id,
                "similarity": round(similarity_score, 3) if matched else "new"
            })

        frame_count += 1

    cap.release()
    pd.DataFrame(log_data).to_csv(OUTPUT_LOG, index=False)
    print(f"\n✅ Character log written to: {OUTPUT_LOG}")
    print("👥 Characters detected:")
    for cid in sorted({row['character_id'] for row in log_data}):
        print(f" - {cid}")

if __name__ == "__main__":
    recognize_characters(VIDEO_PATH)
