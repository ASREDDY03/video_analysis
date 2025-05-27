import os
import pandas as pd
from moviepy.editor import VideoFileClip
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# Paths
SCENE_BOUNDARY_PATH = "debug_outputs/scene_boundaries.csv"
EVENT_PATH = "debug_outputs/event_log.csv"
INTERACTION_PATH = "debug_outputs/interaction_log.csv"
CHARACTER_TRACKING_PATH = "debug_outputs/character_tracking_log.csv"
OUTPUT_CSV = "debug_outputs/test_semantic_scene_narration.csv"

# Load factual summarizer
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Load lightweight BLIP model
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_scene_caption(image_np):
    image_pil = Image.fromarray(image_np).convert("RGB")
    inputs = blip_processor(image_pil, return_tensors="pt").to(blip_model.device)
    output = blip_model.generate(**inputs)
    caption = blip_processor.decode(output[0], skip_special_tokens=True)
    if "image" in caption.lower() or len(caption.strip().split()) < 3:
        return "No clear objects or people visible in this frame."
    return caption

def extract_middle_frame(video_path, start_time, end_time):
    clip = VideoFileClip(video_path).subclip(start_time, end_time)
    return clip.get_frame(clip.duration / 2)

def summarize_scene(scene_id, scene_start, scene_end, video_path, events, interactions, character_df):
    keyframe = extract_middle_frame(video_path, scene_start, scene_end)
    caption = generate_scene_caption(keyframe)

    scene_characters = character_df[
        character_df['frame'].str.extract(r"(\d+)").astype(float)[0].between(scene_start, scene_end)
    ]["characters"].unique().tolist()
    characters = ", ".join(scene_characters) if scene_characters else "unknown individuals"

    prompt = (
        f"Scene {scene_id}: Duration {scene_start:.1f}s to {scene_end:.1f}s.\n"
        f"People detected: {characters}\n"
        f"Keyframe visual: {caption}\n"
        f"Detected events: {events}\n"
        f"Detected interactions: {interactions}\n\n"
        "Write a simple 2-sentence summary of the visual and actions. Do not invent names, emotions, or events not seen in the data."
    )

    result = summarizer(prompt, max_length=150, do_sample=False)[0]['summary_text']
    return {
        "scene_id": scene_id,
        "start_time": scene_start,
        "end_time": scene_end,
        "characters": characters,
        "events": events,
        "interactions": interactions,
        "keyframe_caption": caption,
        "summary": result
    }

def generate_keyframe_test_narration(video_path):
    print("\nRunning keyframe-based scene summarization...")
    if not os.path.exists(SCENE_BOUNDARY_PATH):
        raise FileNotFoundError("Scene boundaries file missing.")

    scene_df = pd.read_csv(SCENE_BOUNDARY_PATH)
    event_df = pd.read_csv(EVENT_PATH) if os.path.exists(EVENT_PATH) else pd.DataFrame(columns=["time_sec", "events"])
    interaction_df = pd.read_csv(INTERACTION_PATH) if os.path.exists(INTERACTION_PATH) else pd.DataFrame(columns=["time_sec", "interactions"])
    character_df = pd.read_csv(CHARACTER_TRACKING_PATH) if os.path.exists(CHARACTER_TRACKING_PATH) else pd.DataFrame(columns=["frame", "characters"])

    narration_results = []
    def time_to_sec(tc): return sum(x * float(t) for x, t in zip([3600, 60, 1], tc.split(":")))

    for i, row in scene_df.iterrows():
        start = time_to_sec(row["scene_start"])
        end = time_to_sec(row["scene_end"])

        events = ", ".join(event_df[event_df.time_sec.between(start, end)]["events"].unique()) or "none"
        interactions = ", ".join(interaction_df[interaction_df.time_sec.between(start, end)]["interactions"].unique()) or "none"

        summary = summarize_scene(i + 1, start, end, video_path, events, interactions, character_df)
        narration_results.append(summary)

    pd.DataFrame(narration_results).to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Test narration saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    test_video_path = "uploads/uploaded_video.mp4"
    generate_keyframe_test_narration(test_video_path)
