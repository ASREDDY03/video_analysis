import os
import cv2
import pandas as pd
from speech_processing import extract_speech
from scene_detection import segment_scenes
from object_detection import detect_objects
from emotion_detection import detect_emotion
from text_summarization import summarize_text

DEBUG_DIR = "debug_frames"  
SRT_OUTPUT_PATH = "debug_outputs/subtitles.srt"  


os.makedirs(DEBUG_DIR, exist_ok=True)
os.makedirs("debug_outputs", exist_ok=True)  
def flatten_list(nested_list):
    """Flattens a nested list into a single list of strings."""
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten_list(item))  # Recursively flatten if it's a nested list
        elif isinstance(item, (str, int, float)):  # Ensure non-list values are added
            flat_list.append(str(item))  # Convert everything to string
    return flat_list

def process_video(video_path):
    """Processes the video by extracting speech, detecting scenes, objects, and emotions."""

    # ✅ Ensure video file exists
    if not os.path.exists(video_path):
        return {"error": " Video file not found!"}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": " Unable to open video file."}

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    print(f"🎥 Processing Video: {video_path} | FPS: {fps} | Total Frames: {total_frames}")

    # ✅ Extract speech & generate subtitles
    transcript, audio_debug_file, srt_file_path = extract_speech(video_path)

    # ✅ Handle case where no speech is detected
    if transcript.strip() in ["", " No speech detected."]:
        transcript = "No speech detected."
        srt_file_path = None  

    # ✅ Detect scene changes
    scene_changes = segment_scenes(video_path)

    # ✅ Process frame-by-frame analysis
    frame_analysis = []
    frame_skip_interval = max(fps // 2, 1)  

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # ✅ Stop processing if no more frames

        if frame_count % frame_skip_interval == 0:  
            object_results = detect_objects(frame)
            emotion_result = detect_emotion(frame)

            # ✅ Ensure `object_results` is a flat list of strings (Fix TypeError)
            flattened_objects = flatten_list(object_results)
            object_text = ", ".join(flattened_objects) if flattened_objects else "None"
            emotion_text = str(emotion_result) if isinstance(emotion_result, str) else "Neutral"

            # ✅ Save frame with classification labels using frame timestamp
            frame_time_sec = int(frame_count / fps)  # ✅ Get frame time in seconds
            frame_filename = f"{DEBUG_DIR}/frame_{frame_time_sec}s.jpg"

            cv2.putText(frame, f"Objects: {object_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Emotion: {emotion_text}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.imwrite(frame_filename, frame)

            frame_analysis.append({
                "frame_time_sec": frame_time_sec,  # ✅ Store frame timestamp (seconds)
                "objects_detected": object_text,
                "facial_emotion": emotion_text,
                "frame_image": frame_filename  # ✅ Store frame filename for reference
            })

        frame_count += 1

    cap.release()

    # ✅ Generate Summary of Transcribed Text
    summary = summarize_text(transcript)

    # ✅ Convert frame analysis to a structured DataFrame
    frame_df = pd.DataFrame(frame_analysis)

    return {
        "speech_summary": summary if summary else "No speech detected.",
        "transcription": transcript,  # ✅ Attach full transcription data
        "srt_file": srt_file_path,  # ✅ Include subtitle file path
        "scene_changes": scene_changes,
        "frame_analysis": frame_df,  # ✅ Include structured DataFrame for table display
        "audio_debug_file": audio_debug_file,  # ✅ Attach extracted audio file for debugging
        "debug_frames": frame_analysis  # ✅ Attach saved frame images with numbers
    }
