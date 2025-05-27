import os
import json
import wave
import subprocess
from vosk import Model, KaldiRecognizer
from resemblyzer import VoiceEncoder, preprocess_wav
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import torch

# CONFIG
VIDEO_PATH = "uploads/uploaded_video.mp4"
AUDIO_PATH = "debug_outputs/audio_extracted.wav"
OUTPUT_JSON_PATH = "debug_outputs/speaker_segments_with_text.json"
VOSK_MODEL_PATH = os.path.expanduser("~/vosk-model")

def extract_audio(video_path, audio_path):
    print("\n🎙️ Extracting audio using ffmpeg...")
    if os.path.exists(audio_path):
        os.remove(audio_path)
        print("⚠️ Old audio file deleted.")

    command = [
        "ffmpeg", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        audio_path
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return os.path.exists(audio_path) and os.path.getsize(audio_path) > 1000

def get_transcript_with_timestamps(audio_path):
    wf = wave.open(audio_path, "rb")
    model = Model(VOSK_MODEL_PATH)
    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)

    words = []
    transcript = []

    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            transcript.append(result.get("text", ""))
            words.extend(result.get("result", []))

    wf.close()
    return words, " ".join(transcript)

def cluster_speakers(audio_path, word_segments, num_speakers=2):
    print("\n🔊 Running Resemblyzer for speaker clustering...")
    encoder = VoiceEncoder()
    wav = preprocess_wav(audio_path)

    segments_audio = []
    timestamps = []

    # Group into phrases with <= 0.5s silence between words
    phrase = []
    window_start = word_segments[0]["start"]

    for i, word in enumerate(word_segments):
        phrase.append(word)
        is_last = i == len(word_segments) - 1
        gap_to_next = (
            word_segments[i + 1]["start"] - word["end"] if not is_last else 1.0
        )

        if gap_to_next > 0.5 or is_last:
            start = phrase[0]["start"]
            end = phrase[-1]["end"]
            text = " ".join(w["word"] for w in phrase)
            segment_wav = wav[int(start * 16000):int(end * 16000)]

            if len(segment_wav) > 0:
                segments_audio.append(segment_wav)
                timestamps.append((start, end, text))
            phrase = []

    embeddings = np.array([encoder.embed_utterance(seg) for seg in segments_audio])
    clustering = AgglomerativeClustering(n_clusters=num_speakers).fit(embeddings)
    labels = clustering.labels_

    speaker_segments = []
    for label, (start, end, text) in zip(labels, timestamps):
        speaker_segments.append({
            "speaker": f"Speaker_{label}",
            "start": round(start, 2),
            "end": round(end, 2),
            "text": text
        })

    return speaker_segments

def merge_segments(speaker_segments, min_duration=1.0):
    if not speaker_segments:
        return []

    merged = [speaker_segments[0]]
    for seg in speaker_segments[1:]:
        last = merged[-1]
        # Merge if same speaker and close in time
        if (
            seg["speaker"] == last["speaker"] and
            seg["start"] - last["end"] < min_duration
        ):
            last["end"] = seg["end"]
            last["text"] += " " + seg["text"]
        else:
            merged.append(seg)
    return merged

def test_audio():
    os.makedirs("debug_outputs", exist_ok=True)

    if not os.path.exists(VIDEO_PATH):
        print("❌ Video file not found:", VIDEO_PATH)
        return

    if not extract_audio(VIDEO_PATH, AUDIO_PATH):
        print("❌ Failed to extract audio.")
        return

    word_segments, _ = get_transcript_with_timestamps(AUDIO_PATH)
    clustered_segments = cluster_speakers(AUDIO_PATH, word_segments, num_speakers=2)
    final_segments = merge_segments(clustered_segments, min_duration=1.0)

    with open(OUTPUT_JSON_PATH, "w") as f:
        json.dump(final_segments, f, indent=2)

    print(f"✅ Speaker segments with transcript saved to: {OUTPUT_JSON_PATH}")

if __name__ == "__main__":
    torch.set_num_threads(1)
    test_audio()