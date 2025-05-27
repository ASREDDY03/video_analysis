import os
import pandas as pd

# Input paths
NARRATION_CSV = "debug_outputs/semantic_scene_narration.csv"
SCENE_LINKS_PATH = "debug_outputs/scene_relationships.csv"

# Output path
CLASSIFIED_LINKS_PATH = "debug_outputs/classified_scene_links.csv"

# Rule-based classification function
def classify_link(row):
    summary_1 = str(row.get('summary_1', '')).lower()
    summary_2 = str(row.get('summary_2', '')).lower()
    score = row.get('similarity_score', 0)

    # Temporal continuation
    if abs(row.get('source_scene', 0) - row.get('target_scene', 0)) == 1 and score > 0.5:
        return "Sequential Continuation"

    # Emotional escalation
    if any(w in summary_2 for w in ["shouting", "crying", "angry", "frustrated"]):
        return "Emotional Escalation"

    # Same characters across scenes
    if any(w in summary_1 for w in ["man", "woman", "child", "person"]) and any(w in summary_2 for w in ["man", "woman", "child", "person"]):
        return "Character Continuation"

    # Visual match
    if any(w in summary_1 for w in ["car", "bridge", "phone"]) and any(w in summary_2 for w in ["car", "bridge", "phone"]):
        return "Object Persistence"

    # Default fallback
    return "Narrative Similarity"

# Main function
def classify_scene_links():
    if not os.path.exists(NARRATION_CSV) or not os.path.exists(SCENE_LINKS_PATH):
        raise FileNotFoundError("Required CSV files missing")

    narration_df = pd.read_csv(NARRATION_CSV)
    if narration_df.shape[0] < 2:
        empty_df = pd.DataFrame(columns=["source_scene", "target_scene", "similarity_score", "summary_1", "summary_2", "link_type"])
        os.makedirs("debug_outputs", exist_ok=True)
        empty_df.to_csv(CLASSIFIED_LINKS_PATH, index=False)
        print("Only one scene detected. Scene classification skipped.")
        print("Scene Summary:")
        return
    link_df = pd.read_csv(SCENE_LINKS_PATH)

    # Merge summaries for both source and target scenes
    merged = link_df.merge(
        narration_df[['scene_id', 'summary']], 
        left_on='source_scene', 
        right_on='scene_id', 
        how='left'
    ).rename(columns={'summary': 'summary_1'}).drop(columns=['scene_id'])

    merged = merged.merge(
        narration_df[['scene_id', 'summary']], 
        left_on='target_scene', 
        right_on='scene_id', 
        how='left'
    ).rename(columns={'summary': 'summary_2'}).drop(columns=['scene_id'])

    # Fill any missing summaries with empty string
    merged['summary_1'] = merged['summary_1'].fillna("")
    merged['summary_2'] = merged['summary_2'].fillna("")

    # Classify the link type
    merged['link_type'] = merged.apply(classify_link, axis=1)

    # Save to CSV
    os.makedirs("debug_outputs", exist_ok=True)
    merged.to_csv(CLASSIFIED_LINKS_PATH, index=False)
    print(f"Scene link types saved to {CLASSIFIED_LINKS_PATH}")

# Standalone execution
if __name__ == "__main__":
    classify_scene_links()
