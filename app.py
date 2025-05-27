import os
import streamlit as st
import subprocess
import zipfile
from run_full_pipeline import run_full_pipeline  #  Replaces `process_video` with modular pipeline
import pandas as pd

try:
    subprocess.run(["python", "cleanup.py"], check=True)
except Exception as e:
    st.warning(f"⚠️ Cleanup script failed: {e}")

# ✅ Set Streamlit page layout
st.set_page_config(layout="wide")
st.title("🎥 Video Transcription & Semantic Understanding")

UPLOAD_DIR = "uploads"
DEBUG_DIR = "debug_frames"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ✅ Upload Section
uploaded_file = st.file_uploader("📤 Upload Video", type=["mp4", "avi", "mov", "mkv"], help="Limit 200MB per file")

if uploaded_file:
    video_path = os.path.join(UPLOAD_DIR, "uploaded_video.mp4")
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("🚀 Running full video analysis pipeline..."):
        results = run_full_pipeline(video_path)

    # ✅ Video and Audio Preview
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("<h3 style='text-align: center;'>▶ Play Video</h3>", unsafe_allow_html=True)
        st.video(video_path)

    with col2:
        if os.path.exists(results.get("audio_debug_file", "")):
            st.markdown("🔊 **Extracted Audio**")
            st.audio(results["audio_debug_file"], format="audio/wav")

    # ✅ Display Transcript Summary
    st.markdown("<h3>📜 Full Transcript </h3>", unsafe_allow_html=True)
    st.text_area("", results.get("speech_summary", "No transcript available."), height=250)

    # ✅ Scene Narration Table
    narration_path = results.get("scene_narration_csv")
    if narration_path and os.path.exists(narration_path):
        narration_df = pd.read_csv(narration_path)
        st.markdown("## 🎬 Scene-by-Scene Semantic Narration")
        for _, row in narration_df.iterrows():
            st.markdown(f"### 🎞️ Scene {int(row['scene_id'])}: `{row['start_time']}` – `{row['end_time']}`")
            st.markdown(f"**📌 Events:** {row['events']}")
            st.markdown(f"**🤝 Interactions:** {row['interactions']}")
            st.markdown(f"**🧠 Summary:** {row['summary']}")
            st.markdown("---")

    # ✅ Subtitle Download
    if os.path.exists(results.get("srt_file", "")):
        with open(results["srt_file"], "rb") as srt_file:
            st.download_button("📥 Download Subtitles (.srt)", data=srt_file, file_name="subtitles.srt", mime="text/plain")

    # ✅ Frame Viewer
    st.markdown("<h3 style='text-align: center;'>📸 Debugging Frames</h3>", unsafe_allow_html=True)
    if os.path.exists(DEBUG_DIR):
        debug_images = sorted(os.listdir(DEBUG_DIR))
        for img in debug_images[:5]:
            st.image(os.path.join(DEBUG_DIR, img), caption=img, use_container_width=True)

        # ZIP Download
        zip_path = os.path.join(DEBUG_DIR, "debug_frames.zip")
        with zipfile.ZipFile(zip_path, "w") as zipf:
            for f in debug_images:
                zipf.write(os.path.join(DEBUG_DIR, f), arcname=f)
        with open(zip_path, "rb") as zip_file:
            st.download_button("📥 Download All Debug Frames", data=zip_file, file_name="debug_frames.zip", mime="application/zip")
