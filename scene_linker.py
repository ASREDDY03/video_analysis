import os
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Paths
SCENE_NARRATION_PATH = "debug_outputs/semantic_scene_narration.csv"
SCENE_LINKS_OUTPUT_PATH = "debug_outputs/scene_relationships.csv"

def compute_scene_relationships():
    if not os.path.exists(SCENE_NARRATION_PATH):
        raise FileNotFoundError(f"Scene narration file not found at {SCENE_NARRATION_PATH}")

    df = pd.read_csv(SCENE_NARRATION_PATH)
    if "summary" not in df.columns:
        raise ValueError("'summary' column not found in semantic_scene_narration.csv")

    summaries = df["summary"].fillna("").tolist()
    scene_ids = df["scene_id"].tolist()

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(summaries, convert_to_tensor=True)

    cosine_sim_matrix = util.cos_sim(embeddings, embeddings)

    related_scenes = []
    for i, sid in enumerate(scene_ids):
        similarities = list(enumerate(cosine_sim_matrix[i]))
        similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
        for j, score in similarities[1:4]:  # Top 3 related scenes
            related_scenes.append({
                "source_scene": sid,
                "target_scene": scene_ids[j],
                "similarity_score": round(score.item(), 3)
            })

    link_df = pd.DataFrame(related_scenes)
    os.makedirs(os.path.dirname(SCENE_LINKS_OUTPUT_PATH), exist_ok=True)
    link_df.to_csv(SCENE_LINKS_OUTPUT_PATH, index=False)
    print(f"Scene relationship links saved to {SCENE_LINKS_OUTPUT_PATH}")

if __name__ == "__main__":
    compute_scene_relationships()
