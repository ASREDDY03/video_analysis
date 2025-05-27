from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
import os
import pandas as pd
import cv2

SCENE_BOUNDARY_PATH = "debug_outputs/scene_boundaries.csv"

def segment_scenes(video_path):
    """Segments video into scenes based on content changes and saves them to a CSV."""
    
    # Ensure video file exists before processing
    if not os.path.exists(video_path):
        return {"error": "Video file not found!"}

    try:
        video_manager = VideoManager([video_path])
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector())

        video_manager.set_downscale_factor()
        video_manager.start()

        scene_manager.detect_scenes(frame_source=video_manager)
        scene_list = scene_manager.get_scene_list()

        # Handle case where no scene changes are detected
        if not scene_list:
            print("⚠️ No scene cuts detected. Using full video duration.")
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            duration_sec = frame_count / fps if fps else 0
            cap.release()

            hours = int(duration_sec // 3600)
            minutes = int((duration_sec % 3600) // 60)
            seconds = int(duration_sec % 60)
            fallback_end = f"{hours:02d}:{minutes:02d}:{seconds:02d}.000"
            scene_ranges = [("00:00:00.000", fallback_end)]
        else:
            scene_ranges = [(start.get_timecode(), end.get_timecode()) for start, end in scene_list]

        # Save to CSV
        os.makedirs("debug_outputs", exist_ok=True)
        df = pd.DataFrame(scene_ranges, columns=["scene_start", "scene_end"])
        df.to_csv(SCENE_BOUNDARY_PATH, index=False)
        print(f" Scene boundaries saved to {SCENE_BOUNDARY_PATH}")

        return scene_ranges

    except Exception as e:
        print(f" Error in scene detection: {e}")
        return {"error": "Scene detection failed due to an error."}
