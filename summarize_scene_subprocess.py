import sys
import json
from transformers import pipeline

def main():
    scene_id = int(sys.argv[1])
    start_time = float(sys.argv[2])
    end_time = float(sys.argv[3])
    caption = sys.argv[4]

    prompt = (
        f"Scene {scene_id}: {start_time:.2f}s to {end_time:.2f}s.\n"
        f"Captions: {caption}\n"
        f"Write a factual and concise summary combining visual and spoken context as a third person describing their understanding after watching the video."
    )

    summarizer = pipeline("text2text-generation", model="google/flan-t5-xl")
    output = summarizer(prompt, max_length=1000, do_sample=False)
    print(json.dumps(output[0]))

if __name__ == "__main__":
    main()
