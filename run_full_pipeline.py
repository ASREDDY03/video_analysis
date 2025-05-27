import os
import pandas as pd
from scene_detection import segment_scenes
from speech_processing import extract_speech
from event_extractor import main as run_event_extractor
from event_interaction_extractor import main as run_interaction_extractor
from scene_narrator import generate_scene_narration
from scene_linker import compute_scene_relationships
from scene_link_classifier import classify_scene_links
from process_video import process_video
from narrative_graph import generate_narrative_graph  
from final_video_summary import generate_video_summary
from speaker_visual_sync import sync_speaker_with_person
from test_char import recognize_characters
from test_audio import test_audio

# Output file paths
DEBUG_OUTPUTS_DIR = "debug_outputs"
TRANSCRIPT_PATH = os.path.join(DEBUG_OUTPUTS_DIR, "transcription.txt")
SRT_PATH = os.path.join(DEBUG_OUTPUTS_DIR, "subtitles.srt")
AUDIO_PATH = os.path.join(DEBUG_OUTPUTS_DIR, "extracted_audio.wav")
SCENE_BOUNDARIES_PATH = os.path.join(DEBUG_OUTPUTS_DIR, "scene_boundaries.csv")
SCENE_NARRATION_PATH = os.path.join(DEBUG_OUTPUTS_DIR, "semantic_scene_narration.csv")
SCENE_RELATIONSHIP_PATH = os.path.join(DEBUG_OUTPUTS_DIR, "scene_relationships.csv")
CLASSIFIED_LINKS_PATH = os.path.join(DEBUG_OUTPUTS_DIR, "classified_scene_links.csv")


def run_full_pipeline(video_path):
    if not os.path.exists(video_path):
        return {"error": "Video file not found!"}


    # Step 1: Scene Detection
    print("\nStep 1: Segmenting scenes...")
    scene_changes = segment_scenes(video_path)
    if isinstance(scene_changes, dict) and "error" in scene_changes:
        return scene_changes
    print(f"Detected {len(scene_changes)} scenes.")

    # Step 2: Speech Transcription
    print("\nStep 2: Extracting speech and transcription...")
    transcript, audio_debug_file, srt_file_path, segments = extract_speech(video_path)
    test_audio()

    # Step 3: Character Detection
    print("\nStep 3: Character Detection...")
    # 3.1 Character Recognition
    print("  3.1 Running Character Recognition...")
    recognize_characters(video_path)

    # 3.2 Speaker-Visual Synchronization
    print("  3.2 Running Speaker-Visual Synchronization...")

    sync_speaker_with_person()

    # 3.3 Role & Name Mapping
    print("  3.3 Running Role and Name Mapping...")
    from role_name_mapper import map_roles_and_names
    map_roles_and_names()


    # Step 3: Frame-Level Analysis
    print("\nStep 3: Analyzing frames for object and emotion detection...")
    video_analysis = process_video(
        video_path,
        scene_changes=scene_changes,
        segments=segments,
        transcript=transcript,
        srt_file_path=srt_file_path,
        audio_debug_file=audio_debug_file
    )
    frame_df = video_analysis["frame_analysis"]
    scene_df = video_analysis["scene_analysis"]

    # Step 4: Motion Event Detection
    print("\nStep 4: Extracting motion-based events...")
    run_event_extractor()

    # Step 5: Interaction Inference
    print("\nStep 5: Inferring object interactions...")
    run_interaction_extractor()

    # Step 6: Scene Narration
    print("\nStep 6: Generating semantic scene narrations...")
    generate_scene_narration()

    # Step 7: Scene Similarity Linking
    print("\nStep 7: Computing scene similarity links...")
    compute_scene_relationships()

    # Step 8: Scene Link Classification
    print("\nStep 8: Classifying scene link types...")
    classify_scene_links()

    # Step 9: Narrative Graph
    print("\nStep 9: Building narrative graph...")
    generate_narrative_graph()
  
    #  Step 10
    print("\nStep 10: Generating final video-level summary...")
    generate_video_summary()

    print("\n✅ Full video understanding pipeline completed.")
    return {
        "transcription": transcript,
        "speech_summary": video_analysis["speech_summary"],
        "srt_file": srt_file_path,
        "audio_debug_file": audio_debug_file,
        "scene_narration_csv": SCENE_NARRATION_PATH,
        "scene_boundaries_csv": SCENE_BOUNDARIES_PATH,
        "scene_relationships_csv": SCENE_RELATIONSHIP_PATH,
        "classified_scene_links_csv": CLASSIFIED_LINKS_PATH,
        "frame_analysis": frame_df,
        "scene_analysis": scene_df
    }


if __name__ == "__main__":
    test_video = "uploads/uploaded_video.mp4"
    results = run_full_pipeline(test_video)
    print(results)