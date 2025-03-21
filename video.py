import cv2
import whisper
import torch
import numpy as np
from deepface import DeepFace
from ultralytics import YOLO
from scenedetect import detect, ContentDetector
from gensim.summarization import summarize

# Load Models
whisper_model = whisper.load_model("base")  # Whisper model for transcription
yolo_model = YOLO("yolov8n.pt")  # YOLOv8 for object detection

# Function to Extract Speech and Convert to Text
def extract_speech(video_path):
    audio_text = whisper_model.transcribe(video_path)
    return audio_text['text']

# Function for Scene Segmentation
def segment_scenes(video_path):
    scene_list = detect(video_path, ContentDetector())
    return scene_list

# Function for Object Detection
def detect_objects(frame):
    results = yolo_model(frame)
    return results.pandas().xyxy[0]  # Bounding box results

# Function for Facial Emotion Detection
def detect_emotion(frame):
    emotions = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
    return emotions[0]['dominant_emotion'] if emotions else "No face detected"

# Function to Process Video
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    transcript = extract_speech(video_path)
    scene_changes = segment_scenes(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Object Detection
        object_results = detect_objects(frame)
        
        # Facial Emotion Recognition
        emotion_result = detect_emotion(frame)
        
        # Display Output
        print(f"Frame {frame_count}: Objects Detected - {object_results['name'].tolist()}")
        print(f"Facial Emotion: {emotion_result}")
        
        frame_count += 1

    cap.release()
    
    # NLP Analysis of Transcription
    summary = summarize(transcript, ratio=0.2)
    
    print("\n=== VIDEO SUMMARY ===")
    print(f"Speech Summary: {summary}")
    print(f"Scene Changes: {scene_changes}")

# Run the Full Video Analysis
video_path = "your_video.mp4"  # Replace with your video file
process_video(video_path)
