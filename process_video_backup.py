import cv2
import torch
import numpy as np
import srt
from deepface import DeepFace
from ultralytics import YOLO
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from transformers import pipeline
from datetime import timedelta
import os

# Set Hugging Face API Token (Replace with your actual token)
HF_TOKEN = "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"

HF_MIRROR = "https://huggingface.co/datasets"
whisper_model = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large",
    token=HF_TOKEN,
    device=0 if torch.cuda.is_available() else -1
)


# Load YOLOv8 for object detection
yolo_model = YOLO("yolov8n.pt")

# Function to Extract Speech and Convert to Text, also save as SRT
def extract_speech(video_path, srt_output_path="output.srt"):
    try:
        result = whisper_model(video_path)
        transcript = result["text"]

        # Convert to SRT format
        subtitles = []
        words = transcript.split()
        start_time = timedelta(seconds=0)
        duration = timedelta(seconds=3)  # Approximate per subtitle
        for index, word in enumerate(words):
            end_time = start_time + duration
            subtitles.append(srt.Subtitle(index, start_time, end_time, word))
            start_time = end_time  # Move start time forward

        # Save SRT file
        with open(srt_output_path, "w", encoding="utf-8") as f:
            f.write(srt.compose(subtitles))

        return transcript, srt_output_path

    except Exception as e:
        print(f"Error in speech extraction: {e}")
        return "", None

# Function to Read and Process an Existing SRT File
def extract_srt(srt_path):
    try:
        with open(srt_path, "r", encoding="utf-8") as f:
            subtitles = list(srt.parse(f.read()))
        return " ".join([sub.content for sub in subtitles])
    except Exception as e:
        print(f"Error reading SRT file: {e}")
        return ""

# Function for Scene Segmentation
def segment_scenes(video_path):
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())
    
    video_manager.set_downscale_factor()
    video_manager.start()
    
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list()
    return [(start.get_timecode(), end.get_timecode()) for start, end in scene_list]

# Function for Object Detection
def detect_objects(frame):
    
    results = yolo_model(frame)
    object_data = results[0].boxes.data.tolist()  # Extract detected objects
    return object_data  # List of detected objects

# Function for Facial Emotion Detection
def detect_emotion(frame):
    try:
        emotions = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        return emotions[0]['dominant_emotion'] if emotions else "No face detected"
    except Exception as e:
        print(f"Error in emotion detection: {e}")
        return "Error"

# Function for Text Summarization
def summarize_text(text):
    if not text:
        return "No speech detected."
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, 3)  # Extract top 3 sentences
    return " ".join([str(sentence) for sentence in summary])

# Function to Process Video
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Unable to open video file."}

    frame_count = 0
    transcript, srt_file = extract_speech(video_path)
    scene_changes = segment_scenes(video_path)
    
    frame_analysis = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % 10 == 0:  # Process every 10th frame for performance
            object_results = detect_objects(frame)
            emotion_result = detect_emotion(frame)

            frame_analysis.append({
                "frame": frame_count,
                "objects_detected": len(object_results),
                "facial_emotion": emotion_result
            })

        frame_count += 1

    cap.release()

    summary = summarize_text(transcript)

    return {
        "speech_summary": summary,
        "scene_changes": scene_changes,
        "frame_analysis": frame_analysis,
        "srt_file": srt_file
    }
