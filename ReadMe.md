# Video Understanding Pipeline

This repository contains a complete pipeline for semantic video understanding. It includes 
scene detection, 
speech transcription, 
object and emotion detection, 
character tracking, 
motion event extraction, 
interaction inference, 
scene narration, 
link classification, and 
final video summarization.



# 1. Unzip the code file/directory 

# 2. Create and Activate Virtual Environment(Bash Commands)

```bash
python3 -m venv env
source env/bin/activate 
```
# 3. Install Dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

# 4. Download Models (Optional)

Models (e.g., YOLOv8) will be automatically downloaded when missing.

# 5 Run the app.py using streamlit it will trigger the entire pipeline 

```bash
streamlit run app.py
```

Upload a video when prompted. The pipeline will execute and display scene narrations, summaries, and outputs.

# Output Structure

debug_outputs/ – CSVs, SRT files, transcription, narration, relationship graph, final summary
debug_frames/` – Frame-level annotated images


These folders are auto-generated on the first run and do not need to be included manually.

## 📌 Project Highlights

- Scene detection with PySceneDetect
- Speech transcription with Vosk
- Object and face tracking with YOLOv8 + DeepSort + DeepFace
- Emotion detection and annotation
- Motion and interaction extraction
- Scene-level summarization and LLM-based narration (supports Hugging Face token)
- Final narrative summary built using graph traversal

##  API Token

For testing of code i have hardcoded the "Hugging Face" Token for regular runs 
The code can be passed through .env file setup if needed when it comes to privacy concerns.
