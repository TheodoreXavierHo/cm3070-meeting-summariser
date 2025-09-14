# core/transcribe.py

"""
Transcribes either an audio or video file to text using Faster-Whisper ASR.
- If input is a video file, extracts audio track first (using ffmpeg).
- Supports .wav, .mp3, .mp4, .mov, .mkv, .avi, .flv, .webm and similar.
- Outputs a timestamped transcript to outputs/transcript.txt.
"""

import sys
import os
import subprocess
import torch
import shutil
from faster_whisper import WhisperModel
from imageio_ffmpeg import get_ffmpeg_exe

def is_video_file(filepath):
    """
    Returns True if the file extension matches a known video format.
    """
    video_exts = {'.mp4', '.mov', '.avi', '.mkv', '.flv', '.webm'}
    return os.path.splitext(filepath)[1].lower() in video_exts

def is_audio_file(filepath):
    """
    Returns True if the file extension matches a known audio format.
    """
    audio_exts = {'.wav', '.mp3', '.aac', '.ogg', '.flac', '.m4a', '.wma'}
    return os.path.splitext(filepath)[1].lower() in audio_exts

def extract_audio_from_video(video_path, audio_path="outputs/tmp_audio.wav"):
    """
    Extract the audio track from a video file using ffmpeg.
    Prefer a portable binary from imageio-ffmpeg; fall back to system ffmpeg.
    Returns the audio file path.
    """

    # Resolve ffmpeg executable
    try:
        from imageio_ffmpeg import get_ffmpeg_exe  # requires imageio-ffmpeg
        ffmpeg = get_ffmpeg_exe()
    except Exception:
        ffmpeg = shutil.which("ffmpeg") or "ffmpeg"  # last-resort fallback

    # Ensure the output directory exists
    out_dir = os.path.dirname(audio_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        ffmpeg,
        "-y",                   # Overwrite without prompting
        "-i", str(video_path),
        "-vn",                  # No video
        "-acodec", "pcm_s16le", # WAV (16-bit PCM)
        "-ar", "16000",         # 16 kHz
        "-ac", "1",             # Mono
        str(audio_path),
    ]

    print(f"Extracting audio via '{ffmpeg}' from {video_path} -> {audio_path}")
    try:
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        # Include a short tail of stderr for easier debugging
        err = (e.stderr or b"").decode("utf-8", errors="ignore")
        tail = err[-800:]  # last ~800 chars
        print(f"[ffmpeg] failed with code {e.returncode}:\n{tail}")
        raise RuntimeError("Audio extraction with ffmpeg failed.") from e

    return audio_path

def transcribe_audio(filepath, output_path="outputs/transcript.txt", model_size="large-v2"):
    """
    Transcribes an audio file using Faster-Whisper.
    Outputs a timestamped transcript to the specified file.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    print(f"Loading Whisper model '{model_size}' on {device}...")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments, info = model.transcribe(filepath, beam_size=5)
    print(f"Detected language: {info.language}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    full_text = ""
    with open(output_path, "w", encoding="utf-8") as f:
        for segment in segments:
            line = f"[{segment.start:.2f}s - {segment.end:.2f}s] {segment.text}\n"
            print(line, end="")
            f.write(line)
            full_text += segment.text + " "
    print(f"Transcript saved to {output_path}")
    return full_text.strip()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python transcribe.py <audio_or_video_file>")
        sys.exit(1)
    inputfile = sys.argv[1]

    # Determine input type
    if is_video_file(inputfile):
        print("Input detected as video file.")
        audiofile = "outputs/tmp_audio.wav"
        extract_audio_from_video(inputfile, audiofile)
        cleanup_audio = True
    elif is_audio_file(inputfile):
        print("Input detected as audio file.")
        audiofile = inputfile
        cleanup_audio = False
    else:
        print("Unsupported file type. Please provide an audio or video file.")
        sys.exit(1)

    # Run transcription
    try:
        transcribe_audio(audiofile)
    except Exception as e:
        print(f"Transcription failed: {e}")
        sys.exit(1)

    # Remove temporary audio file if created
    if cleanup_audio and os.path.exists(audiofile):
        os.remove(audiofile)
        print(f"Removed temporary audio file: {audiofile}")
