import pandas as pd
import os
from itertools import combinations
from math import dist

TRACKING_CSV = "debug_outputs/object_tracking_log.csv"
OUTPUT_CSV = "debug_outputs/interaction_log.csv"
PROXIMITY_THRESHOLD = 75  # pixels

def detect_interactions(df):
    interactions = []

    for time_sec, frame_data in df.groupby("time_sec"):
        frame_events = []
        people = frame_data[frame_data["class"] == "person"]
        others = frame_data[frame_data["class"] != "person"]

        # 👥 Group detection (cluster of people)
        if len(people) >= 3:
            frame_events.append("group walking")

        # 🧍‍♂️ Near other object
        for _, p_row in people.iterrows():
            for _, o_row in others.iterrows():
                p_center = (p_row["center_x"], p_row["center_y"])
                o_center = (o_row["center_x"], o_row["center_y"])
                if dist(p_center, o_center) < PROXIMITY_THRESHOLD:
                    frame_events.append(f"{p_row['class']} near {o_row['class']}")

        # Person to person proximity (could imply talking/interacting)
        for (i, row1), (j, row2) in combinations(people.iterrows(), 2):
            c1 = (row1["center_x"], row1["center_y"])
            c2 = (row2["center_x"], row2["center_y"])
            if dist(c1, c2) < PROXIMITY_THRESHOLD:
                frame_events.append("persons interacting")

        interactions.append({
            "time_sec": time_sec,
            "interactions": ", ".join(set(frame_events)) or "none"
        })

    return pd.DataFrame(interactions)

def main():
    if not os.path.exists(TRACKING_CSV):
        print(" Object tracking CSV not found.")
        return

    df = pd.read_csv(TRACKING_CSV)
    interaction_df = detect_interactions(df)
    interaction_df.to_csv(OUTPUT_CSV, index=False)

    print(f"Interaction log saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
