import os
import shutil
import streamlit as st
import subprocess
import wave
import json
from vosk import Model, KaldiRecognizer

# ✅ Define Paths
UPLOAD_DIR = "uploads"
DEBUG_DIR = "debug_outputs"
MODEL_PATH = os.path.expanduser("~/vosk-model")  # ✅ Ensure this path is correct
FFMPEG_PATH = "/opt/homebrew/bin/ffmpeg"  # ✅ Corrected FFmpeg path

# ✅ Ensure necessary directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)

st.set_page_config(layout="wide")  # ✅ Must be the first Streamlit command
st.title("🔍 Debug Speech Extraction from Video")

# ✅ Upload Section
uploaded_file = st.file_uploader("📤 Upload Video for Speech Debugging", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file:
    # ✅ Save the uploaded video
    video_path = os.path.join(UPLOAD_DIR, "uploaded_video.mp4")
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"✅ Video saved: {video_path}")

    # ✅ Extract Audio from Video
    audio_path = os.path.join(DEBUG_DIR, "extracted_audio.wav")
    st.info("🔄 Extracting audio...")

    try:
        command = [FFMPEG_PATH, "-i", video_path, "-vn", "-ac", "1", "-ar", "16000", "-y", audio_path]
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        # ✅ Verify if audio was extracted successfully
        if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
            st.success(f"✅ Audio extracted: {audio_path}")
            st.audio(audio_path, format="audio/wav")
        else:
            st.error("❌ Audio extraction failed!")

    except Exception as e:
        st.error(f"❌ Audio extraction error: {e}")

    # ✅ Transcribe the Extracted Audio
    transcript = ""
    transcript_path = os.path.join(DEBUG_DIR, "transcription.txt")
    transcript_json_path = os.path.join(DEBUG_DIR, "transcription.json")

    st.info("🔄 Transcribing audio using Vosk...")

    try:
        # ✅ Ensure Vosk model exists
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"❌ Vosk model not found at {MODEL_PATH}. Please check the path.")

        model = Model(MODEL_PATH)
        wf = wave.open(audio_path, "rb")

        # ✅ Ensure valid audio stream
        if wf.getnframes() == 0:
            st.error("❌ Extracted audio contains no valid frames!")
            wf.close()
        else:
            rec = KaldiRecognizer(model, wf.getframerate())
            rec.SetWords(True)  # ✅ Enable word timestamps for debugging

            transcript_data = []

            # ✅ Process each chunk of audio
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    transcript += result["text"] + " "
                    if "result" in result:
                        transcript_data.extend(result["result"])  # ✅ Collect timestamps

            wf.close()

            # ✅ Save transcript to text file
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(transcript.strip() if transcript.strip() else "❌ No speech detected.")

            # ✅ Save word-level timestamps as JSON
            with open(transcript_json_path, "w", encoding="utf-8") as f:
                json.dump(transcript_data, f, indent=4)

            # ✅ Debugging output
            st.success(f"✅ Transcript saved: {transcript_path}")
            st.text_area("📝 Extracted Transcript:", transcript)

            # ✅ Save extracted audio permanently
            final_audio_path = os.path.join("output", "final_extracted_audio.wav")
            os.makedirs("output", exist_ok=True)  # ✅ Ensure output directory exists
            shutil.copy(audio_path, final_audio_path)

            st.success(f"✅ Final Audio Saved: {final_audio_path}")

            # ✅ Download buttons for debugging
            with open(audio_path, "rb") as audio_file:
                st.download_button(label="📥 Download Extracted Audio", data=audio_file, file_name="extracted_audio.wav", mime="audio/wav")

            with open(final_audio_path, "rb") as final_audio_file:
                st.download_button(label="📥 Download Final Audio", data=final_audio_file, file_name="final_extracted_audio.wav", mime="audio/wav")

            with open(transcript_path, "r", encoding="utf-8") as transcript_file:
                st.download_button(label="📄 Download Transcription File", data=transcript_file, file_name="transcription.txt", mime="text/plain")

            with open(transcript_json_path, "r", encoding="utf-8") as json_file:
                st.download_button(label="📄 Download Transcript JSON", data=json_file, file_name="transcription.json", mime="application/json")

    except Exception as e:
        st.error(f"❌ Error in speech extraction: {e}")
