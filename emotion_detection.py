from deepface import DeepFace
from collections import Counter

def detect_emotion(frame):
    """Detects dominant facial emotion in a video frame using DeepFace."""
    try:
        # ✅ Perform emotion analysis
        emotions = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        # ✅ Ensure valid results
        if isinstance(emotions, list) and len(emotions) > 0:
            emotion_list = [entry['dominant_emotion'] for entry in emotions if 'dominant_emotion' in entry]

            if emotion_list:
                # ✅ Return the most common emotion if multiple faces detected
                most_common_emotion = Counter(emotion_list).most_common(1)[0][0]
                return most_common_emotion
            else:
                return "No face detected"
        
        return "No face detected"

    except Exception as e:
        print(f"❌ Error in emotion detection: {e}")
        return "Error"
