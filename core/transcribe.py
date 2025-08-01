# core/transcribe.py
import sys
import torch
from faster_whisper import WhisperModel

def transcribe_audio(filepath, output_path="outputs/transcript.txt"):
    # Adjustable model size ("base", "small", "medium", "large-v2", etc.)
    model_size = "large-v2"
    # Detect GPU if available, else use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = WhisperModel(model_size, device=device, compute_type="float16" if device == "cuda" else "int8")
    segments, info = model.transcribe(filepath, beam_size=5)
    print(f"Detected language: {info.language}")

    full_text = ""
    # Open output file at the start, write as you go
    with open(output_path, "w", encoding="utf-8") as f:
        for segment in segments:
            line = f"[{segment.start:.2f}s - {segment.end:.2f}s] {segment.text}\n"
            print(line, end="")  # also print to console
            f.write(line)
            full_text += segment.text + " "
            # Optionally, flush every 100 segments for very long jobs
            # if segment.id % 100 == 0: f.flush()
    print(f"\nTranscript saved to {output_path}")
    return full_text.strip()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python transcribe.py <audiofile.wav>")
        sys.exit(1)
    audiofile = sys.argv[1]
    transcribe_audio(audiofile)
