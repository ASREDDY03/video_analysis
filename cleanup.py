import os
import shutil

# Define directories to clean
UPLOAD_DIR = "uploads"
DEBUG_DIR = "debug_outputs"
FRAME_DIR = "debug_frames"
OUTPUT_DIR = "output"  
ZIP_FILE_PATH = os.path.join(FRAME_DIR, "debug_frames.zip")

def ensure_directories():
    """Ensures all necessary directories exist."""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(DEBUG_DIR, exist_ok=True)
    os.makedirs(FRAME_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_old_files():
    """Removes old files and clears directories before a new video is uploaded."""
    print("Cleaning old files before new video processing...")

    # Remove all files from the uploads directory
    if os.path.exists(UPLOAD_DIR):
        for file in os.listdir(UPLOAD_DIR):
            file_path = os.path.join(UPLOAD_DIR, file)
            try:
                os.remove(file_path)
                print(f"Deleted uploaded file: {file_path}")
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")

    # Remove all files from the debug_outputs directory
    if os.path.exists(DEBUG_DIR):
        for file in os.listdir(DEBUG_DIR):
            file_path = os.path.join(DEBUG_DIR, file)
            try:
                os.remove(file_path)
                print(f"Deleted debug output: {file_path}")
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")

    # Remove and recreate debug_frames directory
    if os.path.exists(FRAME_DIR):
        shutil.rmtree(FRAME_DIR)
        os.makedirs(FRAME_DIR)

    # Remove all files from the output directory
    if os.path.exists(OUTPUT_DIR):
        for file in os.listdir(OUTPUT_DIR):
            file_path = os.path.join(OUTPUT_DIR, file)
            try:
                os.remove(file_path)
                print(f"Deleted output file: {file_path}")
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")

    print("Cleanup completed. Ready for new video processing!")

# Run cleanup before processing starts
if __name__ == "__main__":
    ensure_directories()
    clean_old_files()
