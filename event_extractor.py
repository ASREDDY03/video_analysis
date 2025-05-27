import pandas as pd
import numpy as np
import os

# Paths
TRACKING_CSV = "debug_outputs/object_tracking_log.csv"
OUTPUT_CSV = "debug_outputs/event_log.csv"

# Thresholds (tweakable for tuning)
MOVEMENT_THRESHOLD = 20       # Pixels to consider an object as moving
CAMERA_PAN_DIRECTION_RATIO = 0.8  # 80%+ objects moving same direction
GROUP_PROXIMITY_THRESHOLD = 70    # For group detection (future use)

def calculate_direction_change(df):
    """Add dx, dy, and movement (Euclidean distance) per tracked object."""
    df = df.sort_values(by=["track_id", "time_sec"])
    df["dx"] = df.groupby("track_id")["center_x"].diff().fillna(0)
    df["dy"] = df.groupby("track_id")["center_y"].diff().fillna(0)
    df["motion"] = np.sqrt(df["dx"]**2 + df["dy"]**2)
    return df

def detect_events(df):
    """Main event detection loop over time frames."""
    events_per_frame = []
    prev_objects = {}

    for time_sec, frame_df in df.groupby("time_sec"):
        current_ids = set(frame_df["track_id"])
        current_frame_events = []

        # Entry and Exit Detection
        new_entries = current_ids - prev_objects.keys()
        exits = prev_objects.keys() - current_ids

        for tid in new_entries:
            obj_cls = frame_df[frame_df["track_id"] == tid]["class"].values[0]
            current_frame_events.append(f"{obj_cls} enters")

        for tid in exits:
            obj_cls = prev_objects[tid]
            current_frame_events.append(f"{obj_cls} exits")

        # Motion Detection
        moving = frame_df[frame_df["motion"] > MOVEMENT_THRESHOLD]
        for _, row in moving.iterrows():
            current_frame_events.append(f"{row['class']} moving")

        # Camera Panning (heuristic based on direction agreement)
        if len(frame_df) >= 3:
            dx_signs = np.sign(frame_df["dx"])
            dy_signs = np.sign(frame_df["dy"])
            if abs(dx_signs.mean()) > CAMERA_PAN_DIRECTION_RATIO:
                current_frame_events.append("camera pans horizontally")
            if abs(dy_signs.mean()) > CAMERA_PAN_DIRECTION_RATIO:
                current_frame_events.append("camera pans vertically")

        # Update previous object states
        prev_objects = {row["track_id"]: row["class"] for _, row in frame_df.iterrows()}

        # Store frame-level result
        events_per_frame.append({
            "time_sec": time_sec,
            "events": ", ".join(set(current_frame_events)) or "none"
        })

    return pd.DataFrame(events_per_frame)

def main():
    if not os.path.exists(TRACKING_CSV):
        print(f"Object tracking file not found: {TRACKING_CSV}")
        return

    df = pd.read_csv(TRACKING_CSV)
    df = calculate_direction_change(df)
    event_df = detect_events(df)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    event_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Event log saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
