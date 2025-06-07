# core/transcribe.py
"""
Usage: python core/transcribe.py path/to/audio.mp3
Saves JSON transcript to outputs/transcript.json
Prints runtime & rough WER placeholder.
"""
import time, sys, json, whisper
from pathlib import Path

audio_path = Path(sys.argv[1])
start = time.time()

model = whisper.load_model("medium")          # ~1.4 GB; fits in 12 GB VRAM
result = model.transcribe(str(audio_path))

Path("outputs").mkdir(exist_ok=True)
json.dump(result, open("outputs/transcript.json", "w"))

print(f"Transcription finished in {time.time()-start:.1f}s")
print(f"Total tokens: {len(result['text'].split())}")
