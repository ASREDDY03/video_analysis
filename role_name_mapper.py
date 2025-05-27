import json
import spacy
import re
import pandas as pd
from collections import defaultdict

# CONFIG
MAPPING_LOG_FILE = "debug_outputs/speaker_person_map_log.json"
TRANSCRIPT_FILE = "debug_outputs/speaker_segments_with_text.json"
CHARACTER_LOG_FILE = "debug_outputs/test_character_log.csv"
OUTPUT_CSV_FILE = "debug_outputs/enriched_character_log.csv"

# Narrator scoring thresholds
MIN_SPEAKING_DURATION = 5.0  # seconds

def load_mapping_log():
    with open(MAPPING_LOG_FILE, "r") as f:
        return json.load(f)

def load_transcript():
    with open(TRANSCRIPT_FILE, "r") as f:
        return json.load(f)

def load_character_log():
    return pd.read_csv(CHARACTER_LOG_FILE)

nlp = spacy.load("en_core_web_sm")

def extract_names_from_transcript(transcript):
    speaker_to_name = {}
    name_patterns = [
        re.compile(r"i[' ]?m (\w+)", re.I),
        re.compile(r"my name is (\w+)", re.I),
        re.compile(r"this is (\w+)", re.I)
    ]
    
    for seg in transcript:
        speaker = seg.get("speaker")
        text = seg.get("text", "")
        if not speaker or not text:
            continue

        # Run POS tagging
        doc = nlp(text)

        for pattern in name_patterns:
            match = pattern.search(text)
            if match:
                possible_name = match.group(1).capitalize()

                # Check if it's a proper noun using spaCy
                for token in doc:
                    if token.text.lower() == possible_name.lower():
                        if token.pos_ == "PROPN":  # Proper noun
                            speaker_to_name[speaker] = possible_name
                        break
                break  # Stop after first match per segment

    return speaker_to_name

def compute_speaking_durations(transcript, mapping_log):
    durations = defaultdict(float)
    for seg in transcript:
        speaker = seg.get("speaker")
        if speaker in mapping_log:
            person = mapping_log[speaker].get("mapped_person")
            if person:
                durations[person] += seg["end"] - seg["start"]
    return durations

def build_person_roles(mapping_log, name_map, durations):
    person_data = {}
    if durations:
        narrator = max(durations, key=durations.get)
    else:
        narrator = None

    for speaker, info in mapping_log.items():
        person_id = info.get("mapped_person")
        if not person_id:
            continue
        is_narrator = person_id == narrator and durations[person_id] >= MIN_SPEAKING_DURATION
        role = "Narrator" if is_narrator else "Supporting Character"
        name = name_map.get(speaker, "None")
        person_data[person_id] = {
            "role": role,
            "name": name
        }
    return person_data

def enrich_character_log(char_df, person_roles):
    char_df["role"] = char_df["character_id"].map(lambda pid: person_roles.get(pid, {}).get("role", "Bystander"))
    char_df["name"] = char_df["character_id"].map(lambda pid: person_roles.get(pid, {}).get("name", "None"))
    return char_df

def map_roles_and_names():
    mapping_log = load_mapping_log()
    transcript = load_transcript()
    name_map = extract_names_from_transcript(transcript)
    durations = compute_speaking_durations(transcript, mapping_log)
    person_roles = build_person_roles(mapping_log, name_map, durations)
    char_df = load_character_log()
    enriched_df = enrich_character_log(char_df, person_roles)
    enriched_df.to_csv(OUTPUT_CSV_FILE, index=False)
    print(f"\n✅ Enriched character log saved to: {OUTPUT_CSV_FILE}")

if __name__ == "__main__":
    map_roles_and_names()