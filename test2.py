# video_scene_summary.py
import os
import gc
import json
import subprocess
import pandas as pd
import numpy as np
from moviepy.editor import VideoFileClip
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import imagehash

# Paths
SCENE_BOUNDARY_PATH = "debug_outputs/scene_boundaries.csv"
EVENT_PATH = "debug_outputs/event_log.csv"
TRANSCRIPT_PATH = "debug_outputs/transcription.txt"
OUTPUT_CSV = "debug_outputs/scene_summary_enhanced.csv"
FINAL_SUMMARY_PATH = "debug_outputs/test_final_video_summary.txt"
TEMP_CAPTION_DIR = "debug_outputs/temp_captions"
os.makedirs(TEMP_CAPTION_DIR, exist_ok=True)

# Load BLIP model once
print("\n⏳ Loading BLIP captioning model...")
blip_processor = BlipProcessor.from_pretrained("saved_models/fine_tuned_blip")
blip_model = BlipForConditionalGeneration.from_pretrained("saved_models/fine_tuned_blip")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
blip_model.to(device)

# Helper: Convert hh:mm:ss to seconds
def time_to_sec(tc):
    return sum(x * float(t) for x, t in zip([3600, 60, 1], tc.split(":")))

# Load transcript
def load_transcript(path):
    return open(path).read() if os.path.exists(path) else ""

# Extract keyframes
def extract_keyframes(video_path, start_time, end_time, num_frames=5):
    try:
        with VideoFileClip(video_path) as full_clip:
            if start_time >= full_clip.duration:
                print(f"⚠️ Scene start {start_time}s beyond video. Skipping.")
                return []
            end_time = min(end_time, full_clip.duration)
            if (end_time - start_time) < 1.0:
                print(f"⚠️ Scene too short. Skipping.")
                return []
            subclip = full_clip.subclip(start_time, end_time)
            timestamps = np.linspace(0, subclip.duration, num=num_frames + 2)[1:-1]
            return [subclip.get_frame(t) for t in timestamps]
    except Exception as e:
        print(f"❌ Frame extraction error: {e}")
        return []

# Hash frame using perceptual hash
def hash_image(frame):
    image = Image.fromarray(frame).convert("RGB")
    return imagehash.phash(image)

# Generate or load cached captions with deduplication
def generate_scene_caption(scene_id, frames):
    caption_path = os.path.join(TEMP_CAPTION_DIR, f"scene_{scene_id}.txt")
    if os.path.exists(caption_path):
        return open(caption_path).read()

    captions = []
    unique_frames = []
    prev_hash = None

    for i, frame in enumerate(frames):
        frame_hash = hash_image(frame)
        if prev_hash is None or abs(frame_hash - prev_hash) > 4:
            unique_frames.append(frame)
            prev_hash = frame_hash
        else:
            print(f"🔁 Frame {i + 1} skipped (duplicate within scene)")

    for i, frame in enumerate(unique_frames):
        image = Image.fromarray(frame).convert("RGB")
        inputs = blip_processor(image, return_tensors="pt").to(device)
        with torch.no_grad():
            output = blip_model.generate(**inputs)
        caption = blip_processor.decode(output[0], skip_special_tokens=True)
        print(f"🖼️ Caption {i + 1}: {caption}")
        captions.append(caption)

    final_caption = " ".join(captions)
    with open(caption_path, "w") as f:
        f.write(final_caption)
    return final_caption

# Subprocess summarizer
def summarize_scene(scene_id, scene_start, scene_end, caption):
    try:
        result = subprocess.run(
            ["python3", "summarize_scene_subprocess.py", str(scene_id), str(scene_start), str(scene_end), caption],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr)
        return json.loads(result.stdout)["generated_text"]
    except Exception as e:
        print(f"❌ Error summarizing Scene {scene_id}: {e}")
        return "[Summary failed]"

# Final summarizer
def generate_final_summary(scene_summaries):
    from transformers import pipeline
    summarizer = pipeline("text2text-generation", model="google/flan-t5-xl")
    combined = " ".join(scene_summaries)
    prompt = f"The following is a sequence of scene-level descriptions from a video:\n{combined}\nWrite a cohesive and informative summary."
    result = summarizer(prompt, max_length=2000, do_sample=False)[0]['generated_text']
    with open(FINAL_SUMMARY_PATH, "w") as f:
        f.write(result)
    print(f"✅ Final video summary saved to: {FINAL_SUMMARY_PATH}")

# Main logic
def generate_enhanced_scene_summaries(video_path):
    if not os.path.exists(SCENE_BOUNDARY_PATH):
        raise FileNotFoundError("Scene boundaries file missing.")

    scene_df = pd.read_csv(SCENE_BOUNDARY_PATH)
    event_df = pd.read_csv(EVENT_PATH) if os.path.exists(EVENT_PATH) else pd.DataFrame(columns=["time_sec", "events"])
    transcript = load_transcript(TRANSCRIPT_PATH)

    results = []
    for i, row in scene_df.iterrows():
        scene_id = i + 1
        start = time_to_sec(row["scene_start"])
        end = time_to_sec(row["scene_end"])
        print(f"\n⏳ Processing Scene {scene_id} from {start:.2f}s to {end:.2f}s")

        events = ", ".join(event_df[event_df.time_sec.between(start, end)]["events"].unique()) or "none"
        transcript_excerpt = transcript[int(start):int(end)][:300] if transcript else ""

        keyframes = extract_keyframes(video_path, start, end)
        if not keyframes:
            continue

        caption = generate_scene_caption(scene_id, keyframes)
        summary = summarize_scene(scene_id, start, end, caption)

        results.append({
            "scene_id": scene_id,
            "start_time": start,
            "end_time": end,
            "caption": caption,
            "events": events,
            "transcript_excerpt": transcript_excerpt,
            "summary": summary
        })

        gc.collect()

    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Scene-level summaries saved to: {OUTPUT_CSV}")
    generate_final_summary([r["summary"] for r in results])

if __name__ == "__main__":
    generate_enhanced_scene_summaries("uploads/uploaded_video.mp4")