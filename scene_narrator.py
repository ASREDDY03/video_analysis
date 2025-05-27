import os
import pandas as pd
from transformers import pipeline

# Paths
EVENT_PATH = "debug_outputs/event_log.csv"
INTERACTION_PATH = "debug_outputs/interaction_log.csv"
TRANSCRIPT_PATH = "debug_outputs/transcription.txt"
SCENE_BOUNDARY_PATH = "debug_outputs/scene_boundaries.csv"
CHARACTER_TRACKING_PATH = "debug_outputs/character_tracking_log.csv"
OUTPUT_CSV = "debug_outputs/semantic_scene_narration.csv"

# Load summarizer model
summarizer = pipeline("text2text-generation", model="google/flan-t5-base")

# Helper to convert HH:MM:SS.mmm to seconds
def parse_timecode(tc):
    h, m, s = tc.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)

# Load transcript text
def load_transcript(path):
    return open(path).read() if os.path.exists(path) else ""

# Load scene boundaries from CSV
def load_scene_boundaries():
    if not os.path.exists(SCENE_BOUNDARY_PATH):
        raise FileNotFoundError("Scene boundaries CSV missing.")
    scene_df = pd.read_csv(SCENE_BOUNDARY_PATH)
    return [(parse_timecode(start), parse_timecode(end)) for start, end in zip(scene_df['scene_start'], scene_df['scene_end'])]

# Assign scene ID based on timestamp
def assign_scene_id(time_sec, scene_boundaries):
    for idx, (start, end) in enumerate(scene_boundaries, start=1):
        if start <= time_sec <= end:
            return idx
    return None

# Merge event and interaction logs
def merge_logs(event_df, interaction_df):
    return pd.merge(event_df, interaction_df, on="time_sec", how="outer").fillna("none")

# Generate scene-level narration
def summarize_scene(scene_id, group_df, scene_start, scene_end, transcript, character_df):
    events = ", ".join(group_df["events"].unique())
    interactions = ", ".join(group_df["interactions"].unique())
    transcript_excerpt = transcript[:500]

    # Extract characters from character tracking log
    scene_characters = character_df[
        
        character_df['frame_time_sec'].astype(str).str.extract(r"(\d+)").astype(float)[0].between(scene_start, scene_end)
    ]["characters"].unique().tolist()
    characters = ", ".join(scene_characters) if scene_characters else "unknown individuals"

    prompt = (
        f"Scene {scene_id} from {scene_start:.2f}s to {scene_end:.2f}s.\n"
        f"Characters: {characters}.\n"
        f"Events: {events}.\n"
        f"Interactions: {interactions}.\n"
        f"Transcript: \"{transcript_excerpt}\"\n"
        f"Write a natural language narration describing what is happening in this scene."
    )

    result = summarizer(prompt, max_length=150, do_sample=True)[0]['generated_text']
    return {
        "scene_id": scene_id,
        "start_time": scene_start,
        "end_time": scene_end,
        "characters": characters,
        "events": events,
        "interactions": interactions,
        "summary": result
    }

# Main pipeline entrypoint
def generate_scene_narration():
    print("Checking required files...")
    for name, path in {
        "EVENT_LOG": EVENT_PATH,
        "INTERACTION_LOG": INTERACTION_PATH,
        "TRANSCRIPT": TRANSCRIPT_PATH,
        "SCENE_BOUNDARIES": SCENE_BOUNDARY_PATH,
        "CHARACTER_LOG": CHARACTER_TRACKING_PATH,
    }.items():
        print(f"{name}: {'Found' if os.path.exists(path) else 'Missing'} ({path})")

    # Load all data sources
    event_df = pd.read_csv(EVENT_PATH)
    interaction_df = pd.read_csv(INTERACTION_PATH)
    transcript = load_transcript(TRANSCRIPT_PATH)
    scene_boundaries = load_scene_boundaries()
    character_df = pd.read_csv(CHARACTER_TRACKING_PATH) if os.path.exists(CHARACTER_TRACKING_PATH) else pd.DataFrame(columns=["frame", "characters"])

    # Assign scene ID to logs
    merged_df = merge_logs(event_df, interaction_df)
    merged_df['scene_id'] = merged_df['time_sec'].apply(lambda t: assign_scene_id(t, scene_boundaries))

    narrations = []
    for scene_id, group in merged_df.groupby("scene_id"):
        if not scene_id:
            continue
        scene_start, scene_end = scene_boundaries[int(scene_id) - 1]
        narr = summarize_scene(scene_id, group, scene_start, scene_end, transcript, character_df)
        narrations.append(narr)

    pd.DataFrame(narrations).to_csv(OUTPUT_CSV, index=False)
    print(f"Scene narration saved to {OUTPUT_CSV}")

# Allow standalone testing
if __name__ == "__main__":
    generate_scene_narration()
