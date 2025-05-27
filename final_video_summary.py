import os
import pandas as pd
import networkx as nx
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# File paths
NARRATION_PATH = "debug_outputs/semantic_scene_narration.csv"
EDGE_PATH = "debug_outputs/narrative_graph_edges.csv"
FINAL_SUMMARY_PATH = "debug_outputs/final_video_summary.txt"

# Load FLAN-T5 model and tokenizer
model_name = "google/flan-t5-xl"  # Change to flan-t5-xxl if system allows
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def load_graph_from_csv():
    if not os.path.exists(EDGE_PATH) or os.path.getsize(EDGE_PATH) == 0:
        print("Graph CSV file is empty or missing. Proceeding without edges.")
        return nx.DiGraph()
    
    try:
        df = pd.read_csv(EDGE_PATH)
        if df.empty:
            print("Graph CSV file is present but contains no edges.")
            return nx.DiGraph()
    except pd.errors.EmptyDataError:
        print("Graph CSV file contains no parsable data. Using empty graph.")
        return nx.DiGraph()

    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(row['source'], row['target'], weight=row.get("score", 0.5))
    return G

def get_ordered_summaries():
    if not os.path.exists(NARRATION_PATH):
        raise FileNotFoundError("Narration file is missing.")

    df = pd.read_csv(NARRATION_PATH)
    G = load_graph_from_csv()

    for scene_id in df['scene_id']:
        if scene_id not in G:
            G.add_node(scene_id)

    try:
        scene_order = list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        print("Graph has cycles or is empty. Falling back to sorted scene order.")
        scene_order = sorted(df['scene_id'].tolist())

    summaries = []
    for sid in scene_order:
        text = df.loc[df['scene_id'] == sid, 'summary'].values
        if len(text) > 0 and isinstance(text[0], str):
            summaries.append(text[0].strip())

    return summaries

def get_cohesive_summary(scene_summaries):
    prompt = (
        "You are a skilled storyteller AI.\n\n"
        "Based on the following scene descriptions, write a single, cohesive narrative that reads like the conclusion of a beautifully crafted film. "
        "It should feel emotionally resonant, cinematic, and complete. Avoid scene numbers or commentary. Instead, tell the story with confidence and poetic flow.\n\n"
        "Here are the scenes:\n"
    )
    for i, text in enumerate(scene_summaries):
        prompt += f"Scene {i + 1}: {text}\n"
    prompt += "\nFinal cohesive story summary:\n"

    # Tokenize and generate summary using FLAN
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    story = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return story.split("Final cohesive story summary:")[-1].strip()

def generate_video_summary():
    scene_summaries = get_ordered_summaries()
    cohesive_story = get_cohesive_summary(scene_summaries)

    with open(FINAL_SUMMARY_PATH, "w") as f:
        f.write(cohesive_story)

    print(f"Final video summary written to: {FINAL_SUMMARY_PATH}")
    return cohesive_story

if __name__ == "__main__":
    generate_video_summary()






"""

V1
import os
import pandas as pd
import requests
import networkx as nx

NARRATION_PATH = "debug_outputs/semantic_scene_narration.csv"
EDGE_PATH = "debug_outputs/narrative_graph_edges.csv"
FINAL_SUMMARY_PATH = "debug_outputs/final_video_summary.txt"

# Hugging Face API config

HF_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"
API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
HEADERS = {
    "Authorization": f"Bearer {HUGGINGFACE_TOKEN}",
    "Content-Type": "application/json"
}

def load_graph_from_csv():
    # Handle empty or missing file gracefully
    if not os.path.exists(EDGE_PATH) or os.path.getsize(EDGE_PATH) == 0:
        print("Graph CSV file is empty or missing. Proceeding without edges.")
        return nx.DiGraph()
    
    try:
        df = pd.read_csv(EDGE_PATH)
        if df.empty:
            print("Graph CSV file is present but contains no edges.")
            return nx.DiGraph()
    except pd.errors.EmptyDataError:
        print("Graph CSV file contains no parsable data. Using empty graph.")
        return nx.DiGraph()

    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(row['source'], row['target'], weight=row.get("score", 0.5))
    return G

def get_ordered_summaries():
    if not os.path.exists(NARRATION_PATH):
        raise FileNotFoundError("Narration file is missing.")

    df = pd.read_csv(NARRATION_PATH)
    G = load_graph_from_csv()

    # Ensure all scene nodes are present
    for scene_id in df['scene_id']:
        if scene_id not in G:
            G.add_node(scene_id)

    # Topological sort if possible
    try:
        scene_order = list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        print("Graph has cycles or is empty. Falling back to sorted scene order.")
        scene_order = sorted(df['scene_id'].tolist())

    summaries = []
    for sid in scene_order:
        text = df.loc[df['scene_id'] == sid, 'summary'].values
        if len(text) > 0 and isinstance(text[0], str):
            summaries.append(text[0].strip())

    return summaries

def get_cohesive_summary(scene_summaries):
    prompt = (
        "You are a skilled storyteller AI.\n\n"
        "Based on the following scene descriptions, write a single, cohesive narrative that reads like the conclusion of a beautifully crafted film. "
        "It should feel emotionally resonant, cinematic, and complete. Avoid scene numbers or commentary. Instead, tell the story with confidence and poetic flow.\n\n"
        "Here are the scenes:\n"
    )
    for i, text in enumerate(scene_summaries):
        prompt += f"Scene {i + 1}: {text}\n"
    prompt += "\nFinal cohesive story summary:\n"

    response = requests.post(API_URL, headers=HEADERS, json={"inputs": prompt, "max_new_tokens": 512})
    if response.status_code == 200:
        return response.json()[0]["generated_text"].split("Final cohesive story summary:")[-1].strip()
    else:
        raise Exception(f"HF API Error {response.status_code}: {response.text}")

def generate_video_summary():
    scene_summaries = get_ordered_summaries()
    cohesive_story = get_cohesive_summary(scene_summaries)

    with open(FINAL_SUMMARY_PATH, "w") as f:
        f.write(cohesive_story)

    print(f"Final video summary written to: {FINAL_SUMMARY_PATH}")
    return cohesive_story

if __name__ == "__main__":
    generate_video_summary()


V2


import os
import pandas as pd
import requests
import networkx as nx

NARRATION_PATH = "debug_outputs/semantic_scene_narration.csv"
EDGE_PATH = "debug_outputs/narrative_graph_edges.csv"
FINAL_SUMMARY_PATH = "debug_outputs/final_video_summary.txt"

# Hugging Face API config
HUGGINGFACE_TOKEN = "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"
HF_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"
API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
HEADERS = {
    "Authorization": f"Bearer {HUGGINGFACE_TOKEN}",
    "Content-Type": "application/json"
}

def load_graph_from_csv():
    df = pd.read_csv("debug_outputs/narrative_graph_edges.csv")
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(row['source'], row['target'], weight=row.get("score", 0.5))
    return G

def get_ordered_summaries():
    if not os.path.exists(NARRATION_PATH) or not os.path.exists(EDGE_PATH):
        raise FileNotFoundError("Required files missing.")

    df = pd.read_csv(NARRATION_PATH)
    G = load_graph_from_csv()

    # Add all scenes even if not connected
    for scene_id in df['scene_id']:
        if scene_id not in G:
            G.add_node(scene_id)

    # Try topological sort; fallback to sorted node list
    try:
        scene_order = list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        scene_order = sorted(G.nodes())

    summaries = []
    for sid in scene_order:
        text = df.loc[df['scene_id'] == sid, 'summary'].values
        if len(text) > 0 and isinstance(text[0], str):
            summaries.append(text[0].strip())

    return summaries

def get_cohesive_summary(scene_summaries):
    #prompt = ("You are a skilled storyteller AI. Based on the following scene descriptions, write a single, cohesive narrative summary that flows naturally, as if retelling a short story:\n\n")
    prompt = (
    "You are a skilled storyteller AI.\n\n"
    "Based on the following scene descriptions, write a single, cohesive narrative that reads like the conclusion of a beautifully crafted film. "
    "It should feel emotionally resonant, cinematic, and complete. Avoid scene numbers or commentary. Instead, tell the story with confidence and poetic flow.\n\n"
    "Here are the scenes:\n"
)
    for i, text in enumerate(scene_summaries):
        prompt += f"Scene {i + 1}: {text}\n"
    prompt += "\nFinal cohesive story summary:\n"

    response = requests.post(API_URL, headers=HEADERS, json={"inputs": prompt, "max_new_tokens": 512})
    if response.status_code == 200:
        return response.json()[0]["generated_text"].split("Final cohesive story summary:")[-1].strip()
    else:
        raise Exception(f"HF API Error {response.status_code}: {response.text}")

def generate_video_summary():
    scene_summaries = get_ordered_summaries()
    cohesive_story = get_cohesive_summary(scene_summaries)

    with open(FINAL_SUMMARY_PATH, "w") as f:
        f.write(cohesive_story)

    print(f"\n✅ Final video summary written to: {FINAL_SUMMARY_PATH}")
    return cohesive_story

if __name__ == "__main__":
    generate_video_summary()
"""
