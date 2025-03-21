from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
import os

def segment_scenes(video_path):
    """Segments video into scenes based on content changes."""
    
    # ✅ Ensure video file exists before processing
    if not os.path.exists(video_path):
        return {"error": "❌ Video file not found!"}

    try:
        video_manager = VideoManager([video_path])
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector())

        video_manager.set_downscale_factor()
        video_manager.start()

        scene_manager.detect_scenes(frame_source=video_manager)
        scene_list = scene_manager.get_scene_list()

        # ✅ Handle case where no scene changes are detected
        if not scene_list:
            return "No scene changes detected"

        return [(start.get_timecode(), end.get_timecode()) for start, end in scene_list]

    except Exception as e:
        print(f"❌ Error in scene detection: {e}")
        return {"error": "❌ Scene detection failed due to an error."}
