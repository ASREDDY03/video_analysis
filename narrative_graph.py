import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util

# Paths
NARRATION_PATH = "debug_outputs/semantic_scene_narration.csv"
GRAPH_IMAGE_PATH = "debug_outputs/narrative_graph.png"
GRAPH_DATA_PATH = "debug_outputs/narrative_graph_edges.csv"

# Load sentence transformer model globally
model = SentenceTransformer("all-MiniLM-L6-v2")

def load_scene_data():
    if not os.path.exists(NARRATION_PATH):
        raise FileNotFoundError(f"Narration file not found at {NARRATION_PATH}")
    return pd.read_csv(NARRATION_PATH)

def compute_embeddings(summaries):
    return model.encode(summaries, convert_to_tensor=True)

def build_graph(df, embeddings, threshold=0.6):
    G = nx.DiGraph()
    scene_ids = df["scene_id"].tolist()

    for idx, row in df.iterrows():
        G.add_node(row["scene_id"], summary=row["summary"])

    for i, sid1 in enumerate(scene_ids):
        for j, sid2 in enumerate(scene_ids):
            if i == j:
                continue
            score = util.cos_sim(embeddings[i], embeddings[j]).item()
            if score > threshold:
                G.add_edge(sid1, sid2, weight=round(score, 3))

    return G

def save_graph_edges(G):
    edges = []
    for u, v, data in G.edges(data=True):
        edges.append({"source": u, "target": v, "score": data["weight"]})
    pd.DataFrame(edges).to_csv(GRAPH_DATA_PATH, index=False)

def visualize_graph(G):
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(12, 8))
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='skyblue')
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=15)
    edge_labels = {(u, v): d["weight"] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title("Narrative Graph: Scene Relationships")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(GRAPH_IMAGE_PATH)
    plt.close()
    print(f"Narrative graph image saved to {GRAPH_IMAGE_PATH}")

def generate_narrative_graph():
    print("Building Narrative Graph...")
    df = load_scene_data()
    # Handle edge case: only one or zero scenes
    if df.shape[0] < 2:
        print("Only one scene detected. Narrative graph generation skipped.")
        pd.DataFrame(columns=["source", "target", "score"]).to_csv(GRAPH_DATA_PATH, index=False)
        return
    summaries = df["summary"].tolist()
    embeddings = compute_embeddings(summaries)
    graph = build_graph(df, embeddings)
    save_graph_edges(graph)
    visualize_graph(graph)
    print(f"Graph saved with {len(graph.nodes)} scenes and {len(graph.edges)} links.")

# Optional for standalone testing
if __name__ == "__main__":
    generate_narrative_graph()
