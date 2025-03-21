import os
import wave
import json
import subprocess
import vosk
from vosk import Model, KaldiRecognizer

DEBUG_DIR = "debug_outputs"
MODEL_PATH = os.path.expanduser("~/vosk-model")
FFMPEG_PATH = "/opt/homebrew/bin/ffmpeg"  

# ✅ Ensure debug directory exists
os.makedirs(DEBUG_DIR, exist_ok=True)

def extract_audio(video_path):
    """Extracts audio from video using FFmpeg and ensures the file is valid."""
    audio_output_path = os.path.join(DEBUG_DIR, "extracted_audio.wav")

    # ✅ Check if FFmpeg exists
    if not os.path.exists(FFMPEG_PATH):
        print(" FFmpeg not found at expected location.")
        return None

    try:
        command = [FFMPEG_PATH, "-i", video_path, "-vn", "-ac", "1", "-ar", "16000", "-y", audio_output_path]
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if not os.path.exists(audio_output_path) or os.path.getsize(audio_output_path) == 0:
            print(" Extracted audio is empty. Retrying extraction with logs...")
            subprocess.run(command, check=True)  # Retry with logging

            if not os.path.exists(audio_output_path) or os.path.getsize(audio_output_path) == 0:
                raise ValueError("Audio extraction failed again.")

        print(f"✅ Audio extracted successfully: {audio_output_path}")
        return audio_output_path

    except Exception as e:
        print(f" Audio extraction error: {e}")
        return None

def generate_srt(transcript_data, srt_file_path):
    """Generates an SRT file from transcript data with timestamps."""
    try:
        with open(srt_file_path, "w", encoding="utf-8") as srt_file:
            index = 1
            for entry in transcript_data:
                start_time = entry.get("start", 0)
                end_time = entry.get("end", start_time + 2)
                text = entry.get("word", "")

                # ✅ Format timestamps for SRT
                start_srt = f"{int(start_time // 3600):02}:{int((start_time % 3600) // 60):02}:{int(start_time % 60):02},000"
                end_srt = f"{int(end_time // 3600):02}:{int((end_time % 3600) // 60):02}:{int(end_time % 60):02},000"

                srt_file.write(f"{index}\n{start_srt} --> {end_srt}\n{text}\n\n")
                index += 1
        print(f"✅ SRT file saved: {srt_file_path}")
    except Exception as e:
        print(f" Error writing SRT file: {e}")

def extract_speech(video_path):
    """Extracts speech from a video using Vosk and generates subtitles."""
    try:
        # ✅ Ensure Vosk model exists before processing
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(" Vosk model not found at the specified path.")

        model = Model(MODEL_PATH)
        audio_path = extract_audio(video_path)

        if not audio_path:
            return "No speech detected.", None, None

        wf = wave.open(audio_path, "rb")

        # ✅ Ensure audio file is not empty
        if wf.getnframes() == 0:
            print(" Extracted audio contains no valid frames!")
            wf.close()
            return " No speech detected.", audio_path, None

        rec = KaldiRecognizer(model, wf.getframerate())
        rec.SetWords(True)

        transcript = ""
        transcript_data = []

        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                transcript += result["text"] + " "
                if "result" in result:
                    transcript_data.extend(result["result"])  # ✅ Store word-level timestamps

        wf.close()

        transcript_file_path = os.path.join(DEBUG_DIR, "transcription.txt")
        with open(transcript_file_path, "w", encoding="utf-8") as f:
            f.write(transcript.strip() if transcript.strip() else " No speech detected.")

        # ✅ Generate & Save SRT File
        srt_file_path = os.path.join(DEBUG_DIR, "subtitles.srt")
        generate_srt(transcript_data, srt_file_path)

        # ✅ Debugging output
        print(f"✅ Transcript saved: {transcript_file_path}")
        print(f"✅ SRT file saved: {srt_file_path}")
        print(f"Transcript Content: {transcript.strip()}")

        return transcript if transcript.strip() else " No speech detected.", audio_path, srt_file_path

    except Exception as e:
        print(f" Error in speech extraction: {e}")
        return " Error processing speech.", None, None
