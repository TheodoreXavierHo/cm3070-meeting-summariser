# core/transcribe.py
import sys
import torch
from faster_whisper import WhisperModel

def transcribe_audio(filepath):
    # You can adjust the model size ("base", "small", "medium", "large-v2", etc.)
    model_size = "medium"
    # Detect GPU if available, else use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = WhisperModel(model_size, device=device, compute_type="float16" if device == "cuda" else "int8")

    segments, info = model.transcribe(filepath, beam_size=5)
    print(f"Detected language: {info.language}")

    full_text = ""
    for segment in segments:
        print(f"[{segment.start:.2f}s - {segment.end:.2f}s] {segment.text}")
        full_text += segment.text + " "
    return full_text.strip()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python transcribe.py <audiofile.wav>")
        sys.exit(1)
    audiofile = sys.argv[1]
    transcript = transcribe_audio(audiofile)
    # Save transcript to file
    with open("outputs/transcript.txt", "w", encoding="utf-8") as f:
        f.write(transcript)
    print("Transcript saved to outputs/transcript.txt")
