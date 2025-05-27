import os
import cv2
import pandas as pd
from collections import Counter
from ultralytics import YOLO
from text_summarization import summarize_text
from object_tracking import run_object_tracking
from emotion_detection import detect_emotion
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

DEBUG_DIR = "debug_frames"
os.makedirs(DEBUG_DIR, exist_ok=True)
os.makedirs("debug_outputs", exist_ok=True)

# Load YOLOv8 model
yolo_model = YOLO("yolov8n.pt")
BATCH_SIZE = 16
SIMILARITY_THRESHOLD = 0.4
CROWD_THRESHOLD = 5

# Initialize known characters list
known_characters = []
character_id_counter = 1

def flatten_list(nested_list):
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten_list(item))
        elif isinstance(item, (str, int, float)):
            flat_list.append(str(item))
    return flat_list

def assign_scene_id(frame_time_sec, scene_seconds):
    for i, (start, end) in enumerate(scene_seconds, start=1):
        if start <= frame_time_sec <= end:
            return i
    return None

def process_video(video_path, scene_changes, segments, transcript, srt_file_path, audio_debug_file):
    global known_characters, character_id_counter

    if not os.path.exists(video_path):
        return {"error": "Video file not found!"}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Unable to open video file."}

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f"\n🎥 Processing Video: {video_path} | FPS: {fps} | Total Frames: {total_frames}")

    # Pre-compute scene seconds
    scene_seconds = []
    for start, end in scene_changes:
        h1, m1, s1 = start.split(":")
        h2, m2, s2 = end.split(":")
        start_sec = int(h1) * 3600 + int(m1) * 60 + float(s1)
        end_sec = int(h2) * 3600 + int(m2) * 60 + float(s2)
        scene_seconds.append((start_sec, end_sec))

    run_object_tracking(video_path)

    frame_analysis = []
    frame_buffer = []
    timestamps = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_time_sec = int(frame_count / fps)
        frame_buffer.append(frame)
        timestamps.append(frame_time_sec)

        if len(frame_buffer) == BATCH_SIZE or frame_count == total_frames - 1:
            results = yolo_model(frame_buffer)

            for i, result in enumerate(results):
                class_ids = result.boxes.cls.tolist() if hasattr(result, 'boxes') else []
                object_names = [result.names[int(cls)] for cls in class_ids]
                flattened_objects = flatten_list(object_names)
                object_text = ", ".join(flattened_objects) if flattened_objects else "None"

                emotion_result = detect_emotion(frame_buffer[i]) if i % 3 == 0 else ""
                emotion_text = str(emotion_result) if isinstance(emotion_result, str) else "Neutral"

                # Character tracking
                character_label = "Detection Failed"
                try:
                    faces = DeepFace.analyze(frame_buffer[i], actions=["embedding"], enforce_detection=False)
                    if not isinstance(faces, list):
                        faces = [faces]

                    detected_ids = []
                    for face in faces:
                        embedding = face.get("embedding")
                        if embedding is None or not isinstance(embedding, list):
                            continue
                        embedding = np.array(embedding).reshape(1, -1)

                        matched = False
                        for known in known_characters:
                            similarity = cosine_similarity(embedding, [known["embedding"]])[0][0]
                            if similarity >= SIMILARITY_THRESHOLD:
                                detected_ids.append(known["id"])
                                matched = True
                                break

                        if not matched:
                            new_id = f"Person_{character_id_counter}"
                            character_id_counter += 1
                            known_characters.append({"id": new_id, "embedding": embedding.flatten()})
                            detected_ids.append(new_id)

                    if detected_ids:
                        character_label = ", ".join(detected_ids) if len(detected_ids) < CROWD_THRESHOLD else "Crowd"

                except Exception as e:
                    pass  # Character detection failure already handled by default

                # Save frame with overlay
                frame_filename = f"{DEBUG_DIR}/frame_{timestamps[i]}s.jpg"
                frame = frame_buffer[i]
                cv2.putText(frame, f"Objects: {object_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Emotion: {emotion_text}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(frame, f"Characters: {character_label}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imwrite(frame_filename, frame)

                scene_id = assign_scene_id(timestamps[i], scene_seconds)

                frame_analysis.append({
                    "frame_time_sec": timestamps[i],
                    "scene_id": scene_id,
                    "objects_detected": object_text,
                    "facial_emotion": emotion_text,
                    "frame_image": frame_filename,
                    "characters": character_label
                })

            frame_buffer.clear()
            timestamps.clear()

        frame_count += 1

    cap.release()

    # Aggregate per scene
    scene_summary = []
    for idx, (start, end) in enumerate(scene_seconds, start=1):
        scene_frames = [f for f in frame_analysis if start <= f["frame_time_sec"] <= end]
        obj_counter = Counter()
        emo_counter = Counter()
        for f in scene_frames:
            obj_counter.update(f["objects_detected"].split(", "))
            if f["facial_emotion"]:
                emo_counter[f["facial_emotion"]] += 1

        top_ids = [obj for obj, _ in obj_counter.most_common(3)]
        top_objects = ", ".join(top_ids)
        dominant_emotion = emo_counter.most_common(1)[0][0] if emo_counter else "Unknown"

        scene_text = " ".join(seg["text"] for seg in segments if start <= seg["start"] <= end) or "No speech in this scene."
        summary = summarize_text(scene_text)

        scene_summary.append({
            "scene_start": start,
            "scene_end": end,
            "top_objects": top_objects,
            "emotion": dominant_emotion,
            "summary": summary
        })

    scene_df = pd.DataFrame(scene_summary)
    frame_df = pd.DataFrame(frame_analysis)

    # ✅ Save character tracking info
    character_df = frame_df[["frame_time_sec", "scene_id", "characters"]]
    character_df.to_csv("debug_outputs/character_tracking_log.csv", index=False)

    return {
        "speech_summary": summarize_text(transcript) if transcript else "No speech detected.",
        "transcription": transcript,
        "srt_file": srt_file_path,
        "scene_changes": scene_changes,
        "frame_analysis": frame_df,
        "scene_analysis": scene_df,
        "audio_debug_file": audio_debug_file,
        "debug_frames": frame_analysis
    }