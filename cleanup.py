import os
import shutil

# ✅ Define directories to clean
CACHE_FILES = [
    "debug_outputs/extracted_audio.wav",
    "debug_outputs/subtitles.srt",
    "debug_outputs/transcription.txt",
    "debug_outputs/transcription.json"
]

UPLOAD_DIR = "uploads"
DEBUG_DIR = "debug_outputs"
FRAME_DIR = "debug_frames"
OUTPUT_DIR = "output"  # ✅ Ensure `output/` folder is also cleared
ZIP_FILE_PATH = os.path.join(FRAME_DIR, "debug_frames.zip")  # ✅ Save ZIP inside `debug_frames/`

def ensure_directories():
    """Ensures all necessary directories exist."""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(DEBUG_DIR, exist_ok=True)
    os.makedirs(FRAME_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_old_files():
    """Removes old files and clears directories before a new video is uploaded."""
    print("🔄 Cleaning old files before new video processing...")

    # ✅ Remove cached files
    for file in CACHE_FILES:
        try:
            if os.path.exists(file):
                os.remove(file)
                print(f"🗑️ Deleted: {file}")
        except Exception as e:
            print(f"❌ Failed to delete {file}: {e}")

    # ✅ Remove all files from the uploads directory
    if os.path.exists(UPLOAD_DIR):
        for file in os.listdir(UPLOAD_DIR):
            file_path = os.path.join(UPLOAD_DIR, file)
            try:
                os.remove(file_path)
                print(f"🗑️ Deleted uploaded file: {file_path}")
            except Exception as e:
                print(f"❌ Failed to delete {file_path}: {e}")

    # ✅ Remove all files from the debug frames directory (including ZIP file)
    if os.path.exists(FRAME_DIR):
        shutil.rmtree(FRAME_DIR)  # ✅ Deletes the entire debug directory
        os.makedirs(FRAME_DIR)  # ✅ Recreate it to avoid errors

    # ✅ Remove all files from the output directory
    if os.path.exists(OUTPUT_DIR):
        for file in os.listdir(OUTPUT_DIR):
            file_path = os.path.join(OUTPUT_DIR, file)
            try:
                os.remove(file_path)
                print(f"🗑️ Deleted output file: {file_path}")
            except Exception as e:
                print(f"❌ Failed to delete {file_path}: {e}")

    print("✅ Cleanup completed. Ready for new video processing!")

# ✅ Run cleanup before processing starts
if __name__ == "__main__":
    ensure_directories()
    clean_old_files()
