import streamlit as st
import tempfile
import os
import torch
import cv2
import numpy as np
from transformers import BlipProcessor, BlipForConditionalGeneration, T5Tokenizer, T5ForConditionalGeneration
from PIL import Image
from typing import List

# Load BLIP model (only once)
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
device = "cuda" if torch.cuda.is_available() else "cpu"
blip_model = blip_model.to(device)

# Load FLAN-T5-XL model and tokenizer
flan_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
flan_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl").to(device)

# List of allowed genre outputs
ALLOWED_GENRES = [
    "Advertisement", "Travel Vlog", "Movie Scene",
    "Gaming Footage", "Pet Video", "Animal Documentary",
    "Educational Video", "Other"
]

# Simple caption generator using BLIP
def generate_caption(frame):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = blip_processor(image, return_tensors="pt").to(device)
    out = blip_model.generate(**inputs)
    caption = blip_processor.decode(out[0], skip_special_tokens=True)
    return caption

# Build a smart prompt for FLAN-T5
def build_genre_prompt(captions: List[str]) -> str:
    prompt = (
        "Here are some scene descriptions from a video:\n\n"
    )
    prompt += "\n".join(f"- {caption}" for caption in captions)
    prompt += (
        "\n\nBased on these scenes, classify the genre of the video. "
        "genre examples: Advertisement, Travel Vlog, Movie Scene, Gaming Footage, Pet Video, Animal Documentary, Educational Video, or any genere that is used in the real world basis.\n"
        "Answer with only the genre name."
    )
    return prompt

# Classify genre using FLAN-T5-XL
def classify_genre_with_llm(captions: List[str]) -> str:
    prompt = build_genre_prompt(captions)

    inputs = flan_tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).to(device)
    output = flan_model.generate(**inputs, max_length=30)
    predicted_text = flan_tokenizer.decode(output[0], skip_special_tokens=True).strip()

    # Post-process output: Match to allowed genres
    for genre in ALLOWED_GENRES:
        if genre.lower() in predicted_text.lower():
            return genre

    # Fallback if not matching
    return "Other"

# Streamlit app
st.title("🎬 Smart Video Genre Predictor (BLIP + FLAN-T5-XL)")

uploaded_file = st.file_uploader("Upload a Video", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    st.video(uploaded_file)

    st.info("🔍 Extracting frames and generating captions... Please wait.")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    st.write(f"Video Duration: {duration:.2f} seconds")

    # Read frames every 1 second
    frame_interval = int(fps)  # 1 frame per second
    frame_count = 0
    captions = []

    while cap.isOpened() and frame_count * frame_interval < total_frames:
        frame_id = frame_count * frame_interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if not ret:
            break

        # Generate caption
        caption = generate_caption(frame)
        captions.append(caption)

        frame_count += 1

        if frame_count >= 10:  # Limit to 10 frames for speed
            break

    cap.release()

    if captions:
        st.success("✅ Captions generated!")
        st.write("Here are the generated scene descriptions:")
        for cap_txt in captions:
            st.write(f"- {cap_txt}")

        st.info("🧠 Predicting genre dynamically using FLAN-T5-XL...")
        predicted_genre = classify_genre_with_llm(captions)

        st.success(f"🎯 Predicted Genre: **{predicted_genre}**")

    os.unlink(video_path)
