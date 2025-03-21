import os
import streamlit as st
import subprocess
import zipfile
from process_video import process_video

# ✅ Run cleanup script at the start
try:
    subprocess.run(["python", "cleanup.py"], check=True)
except Exception as e:
    st.warning(f"⚠️ Cleanup script failed: {e}")

# ✅ Set Streamlit page layout
st.set_page_config(layout="wide")
st.title("🎥 Video Transcription & Analysis")

UPLOAD_DIR = "uploads"
DEBUG_DIR = "debug_frames"

# ✅ Upload Section
uploaded_file = st.file_uploader("📤 Upload Video", type=["mp4", "avi", "mov", "mkv"], help="Limit 200MB per file")

if uploaded_file:
    # ✅ Save the uploaded video inside `uploads/`
    video_path = os.path.join(UPLOAD_DIR, "uploaded_video.mp4")
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())


    with st.spinner("Processing video... Please wait."):
        results = process_video(video_path)

   
    col1, col2 = st.columns([2, 1]) 

    with col1:
        if os.path.exists(video_path):
            st.markdown("<h3 style='text-align: center;'>▶ Play Video</h3>", unsafe_allow_html=True)
            st.video(video_path)
        else:
            st.error(" Error: The video file could not be found.")

    with col2:
        if "audio_debug_file" in results and results["audio_debug_file"] and os.path.exists(results["audio_debug_file"]):
            st.markdown("🔊 **Extracted Audio Playback**")
            st.audio(results["audio_debug_file"], format="audio/wav")

        if "final_audio_file" in results and results["final_audio_file"] and os.path.exists(results["final_audio_file"]):
            st.markdown("🎵 **Final Processed Audio Playback**")
            st.audio(results["final_audio_file"], format="audio/wav")

    # ✅ Display Transcription with Scrollable Box
    st.markdown("<h3>📜 Extracted Transcript:</h3>", unsafe_allow_html=True)

    transcript_text = results.get("speech_summary", "No transcript available.")
    st.text_area("", transcript_text, height=250)  # ✅ Scrollable transcript window

    # ✅ Display Object Detection & Emotion Analysis as a Table
    st.markdown("<h3 style='text-align: center;'>📊 Object Detection & Emotion Analysis</h3>", unsafe_allow_html=True)
    st.dataframe(results["frame_analysis"])

    # ✅ Attach Subtitle File (if available)
    if "srt_file" in results and results["srt_file"] and os.path.exists(results["srt_file"]):
        st.markdown(f"<h3 style='text-align: center;'>🔤 Subtitles File</h3>", unsafe_allow_html=True)
        with open(results["srt_file"], "rb") as srt_file:
            st.download_button(label="📥 Download Subtitles", data=srt_file, file_name="subtitles.srt", mime="text/plain")
    else:
        st.warning("⚠️ No subtitles file available.")

    # ✅ Debugging Section - Download Extracted Audio & Transcript
    for key, debug_file in results.items():
        if key in ["audio_debug_file", "final_audio_file", "transcription_debug_file"] and debug_file and os.path.exists(debug_file):
            with open(debug_file, "rb") as file:
                st.download_button(label=f"📥 Download {key.replace('_', ' ').title()}", data=file, file_name=os.path.basename(debug_file), mime="text/plain")
        elif key in ["audio_debug_file", "final_audio_file", "transcription_debug_file"] and not debug_file:
            st.warning(f"⚠️ {key.replace('_', ' ').title()} not found.")

    # ✅ Debugging Frames - Save and Download ZIP Inside `debug_frames/`
    st.markdown("<h3 style='text-align: center;'>📸 Debugging Frames</h3>", unsafe_allow_html=True)

    frame_folder = "debug_frames/"
    zip_file_path = os.path.join(frame_folder, "debug_frames.zip")  # ✅ Save ZIP in `debug_frames/`

    if os.path.exists(frame_folder) and os.listdir(frame_folder):
        frame_files = sorted(os.listdir(frame_folder))

        # ✅ Show first 5 images for preview
        for frame_file in frame_files[:5]:
            frame_path = os.path.join(frame_folder, frame_file)
            st.image(frame_path, caption=f"{frame_file}", use_container_width=True)

        # ✅ Create ZIP file for all frames inside `debug_frames/`
        try:
            with zipfile.ZipFile(zip_file_path, "w") as zipf:
                for frame_file in frame_files:
                    zipf.write(os.path.join(frame_folder, frame_file), arcname=frame_file)

            # ✅ Add ZIP Download Button
            with open(zip_file_path, "rb") as zip_file:
                st.download_button(label="📥 Download All Frames", data=zip_file, file_name="debug_frames.zip", mime="application/zip")
        except Exception as e:
            st.error(f"❌ Error creating ZIP file: {e}")
    else:
        st.warning("⚠️ No frames available for debugging.")
