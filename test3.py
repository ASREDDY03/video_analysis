import os
import pandas as pd
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

# Paths
INPUT_CSV = "debug_outputs/scene_summary_enhanced.csv"
OUTPUT_CSV = "debug_outputs/testing.csv"

# Load models
print("⏳ Loading models...")
summarizer = pipeline("text2text-generation", model="t5-large")
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # Lightweight, fast

# Improved caption deduplication using cosine similarity
def deduplicate_captions(captions, threshold=0.88):
    if len(captions) <= 1:
        return captions

    embeddings = embedder.encode(captions, convert_to_tensor=True)
    keep = []
    used = set()

    for i, caption in enumerate(captions):
        if i in used:
            continue
        keep.append(caption)
        for j in range(i + 1, len(captions)):
            if j in used:
                continue
            similarity = util.pytorch_cos_sim(embeddings[i], embeddings[j]).item()
            if similarity >= threshold:
                used.add(j)
    return keep

# Generate summaries for each scene
def generate_summaries_from_captions(input_csv, output_csv):
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input file '{input_csv}' not found.")

    df = pd.read_csv(input_csv)
    results = []

    for _, row in df.iterrows():
        scene_id = row["scene_id"]
        start_time = row["start_time"]
        end_time = row["end_time"]
        raw_caption_text = str(row["caption"]).strip()

        captions = [c.strip() for c in raw_caption_text.split('.') if len(c.strip()) > 3]
        if not captions:
            print(f"⚠️ Scene {scene_id} has no valid captions.")
            continue

        unique_captions = deduplicate_captions(captions)
        if not unique_captions:
            print(f"⚠️ Scene {scene_id} captions were all duplicates.")
            continue

        prompt = (
            f"The following are visual captions from images in a video scene:\n"
            f"{'; '.join(unique_captions)}\n"
            f"Using this information, write a single coherent summary of what is happening in the scene."
        )

        try:
            output = summarizer(prompt, max_length=200, do_sample=False)[0]['generated_text']
        except Exception as e:
            print(f"❌ Failed to summarize Scene {scene_id}: {e}")
            output = "[Summary failed]"

        results.append({
            "scene_id": scene_id,
            "start_time": start_time,
            "end_time": end_time,
            "generated_summary": output.strip()
        })

    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"✅ Scene summaries written to {output_csv}")

if __name__ == "__main__":
    generate_summaries_from_captions(INPUT_CSV, OUTPUT_CSV)
