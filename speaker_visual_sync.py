import json
import pandas as pd
from collections import defaultdict

# CONFIG
SPEAKER_SEGMENTS_PATH = "debug_outputs/speaker_segments_with_text.json"
CHARACTER_LOG_PATH = "debug_outputs/test_character_log.csv"
OUTPUT_MAP_PATH = "debug_outputs/speaker_person_map.json"


def load_speaker_segments():
    with open(SPEAKER_SEGMENTS_PATH, "r") as f:
        return json.load(f)


def load_character_tracking():
    df = pd.read_csv(CHARACTER_LOG_PATH)
    return df[df['character_id'].notnull() & (df['character_id'] != "Detection Failed")]


def align_speakers_to_persons(speaker_segments, char_df, min_visible_frames=3, min_similarity=0.6):
    speaker_to_person_scores = defaultdict(lambda: defaultdict(float))
    speaker_to_person_frames = defaultdict(lambda: defaultdict(list))

    for segment in speaker_segments:
        speaker = segment['speaker']
        start, end = segment['start'], segment['end']

        segment_chars = char_df[(char_df['time_sec'] >= start) & (char_df['time_sec'] <= end)]

        for _, row in segment_chars.iterrows():
            person_id = row['character_id']
            similarity = row.get('similarity', 1.0)
            try:
                similarity = float(similarity)
            except:
                continue

            if similarity >= min_similarity:
                speaker_to_person_scores[speaker][person_id] += similarity
                speaker_to_person_frames[speaker][person_id].append(round(row['time_sec'], 2))

    final_map = {}
    detailed_log = {}

    all_speakers = {seg['speaker'] for seg in speaker_segments}

    for speaker in all_speakers:
        scores = speaker_to_person_scores.get(speaker, {})
        if scores:
            best_person = max(scores, key=scores.get)
            visible_times = speaker_to_person_frames[speaker][best_person]
            score = round(scores[best_person], 2)

            if len(visible_times) >= min_visible_frames:
                final_map[speaker] = best_person
                detailed_log[speaker] = {
                    "mapped_person": best_person,
                    "score": score,
                    "visible_times": visible_times
                }
            else:
                final_map[speaker] = None
                detailed_log[speaker] = {
                    "mapped_person": None,
                    "score": score,
                    "visible_times": visible_times,
                    "reason": f"Not enough valid frames ({len(visible_times)} < min {min_visible_frames})"
                }
        else:
            final_map[speaker] = None
            detailed_log[speaker] = {
                "mapped_person": None,
                "score": 0,
                "visible_times": [],
                "reason": "No character detected during speaking time"
            }

    return final_map, detailed_log


def save_mapping(mapping, log):
    with open(OUTPUT_MAP_PATH, "w") as f:
        json.dump(mapping, f, indent=2)
    log_path = OUTPUT_MAP_PATH.replace(".json", "_log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\n✅ Speaker-to-person mapping saved to {OUTPUT_MAP_PATH}")
    print(f"📋 Detailed log saved to {log_path}")


def sync_speaker_with_person():
    speaker_segments = load_speaker_segments()
    char_df = load_character_tracking()
    mapping, log = align_speakers_to_persons(speaker_segments, char_df)
    save_mapping(mapping, log)


if __name__ == "__main__":
    sync_speaker_with_person()
