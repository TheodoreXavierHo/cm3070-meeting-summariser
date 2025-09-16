# core/transcribe.py

"""
Transcribes either an audio or video file to text using Faster-Whisper ASR.
- If input is a video file, extracts audio track first (using ffmpeg).
- Supports .wav, .mp3, .mp4, .mov, .mkv, .avi, .flv, .webm and similar.
- Outputs a timestamped transcript to outputs/transcript.txt.

Iterative update:
- Auto model switching based on available GPU memory.
- Graceful fallbacks on CUDA OOM: degrade compute_type and/or model size.
- Clear logging of the selected ASR config and fallbacks taken.
"""

import sys
import os
import subprocess
import torch
import shutil
from faster_whisper import WhisperModel
from imageio_ffmpeg import get_ffmpeg_exe  # portable ffmpeg locator

# -------------------------- helpers: file type detection -------------------------- #

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

# ------------------------------ ffmpeg audio extract ------------------------------ #

def extract_audio_from_video(video_path, audio_path="outputs/tmp_audio.wav"):
    """
    Extract the audio track from a video file using ffmpeg.
    Prefer a portable binary from imageio-ffmpeg; fall back to system ffmpeg.
    Returns the audio file path.
    """

    # Resolve ffmpeg executable
    ffmpeg = None
    try:
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
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        # Include a short tail of stderr for easier debugging
        err = (e.stderr or b"").decode("utf-8", errors="ignore")
        tail = err[-800:]  # last ~800 chars
        print(f"[ffmpeg] failed with code {e.returncode}:\n{tail}")
        raise RuntimeError("Audio extraction with ffmpeg failed.") from e

    return audio_path

# --------------------------- ASR config & fallbacks --------------------------- #

def _gpu_vram_gb():
    """
    Return total VRAM (GiB) for the current CUDA device, or 0.0 if CUDA not available.
    """
    if not torch.cuda.is_available():
        return 0.0
    try:
        props = torch.cuda.get_device_properties(0)
        return float(props.total_memory) / (1024 ** 3)
    except Exception:
        return 0.0

def _initial_asr_config(preferred_model="large-v2"):
    """
    Choose an initial (model_size, device, compute_type) based on GPU VRAM.
    We prefer quality, then back off for smaller VRAM.
    """
    vram = _gpu_vram_gb()
    has_cuda = vram > 0.0

    # Allow override via env if the user really wants to force a model.
    forced = os.getenv("FYP_ASR_MODEL", "").strip()
    if forced:
        # Respect forced model but still pick sane device/compute.
        if has_cuda:
            compute = "float16" if vram >= 12.0 else "int8_float16"
            return (forced, "cuda", compute)
        else:
            return (forced, "cpu", "int8_float32")

    if has_cuda:
        # Heuristic thresholds for Faster-Whisper:
        #  - >=12 GiB : large-v2 float16
        #  - 8–12 GiB : large-v2 int8_float16
        #  - 5–8  GiB : medium int8_float16
        #  - 3–5  GiB : small  int8_float16
        if vram >= 12.0:
            return (preferred_model, "cuda", "float16")
        elif vram >= 8.0:
            return (preferred_model, "cuda", "int8_float16")
        elif vram >= 5.0:
            return ("medium", "cuda", "int8_float16")
        elif vram >= 3.0:
            return ("small", "cuda", "int8_float16")
        else:
            return ("small", "cpu", "int8_float32")
    else:
        # CPU defaults: keep accuracy with reasonable speed
        return ("small", "cpu", "int8_float32")

def _asr_fallback_chain(model_size, device, compute_type):
    """
    Generate a sequence of increasingly lighter configs to try if we hit CUDA OOM or init/transcribe errors.
    Starts with the provided triple, then backs off.

    Typical chain (GPU):
      large-v2 float16 -> large-v2 int8_float16 -> medium int8_float16 ->
      small int8_float16 -> small (CPU) int8_float32

    Typical chain (CPU):
      small int8_float32 -> base int8_float32
    """
    chain = [(model_size, device, compute_type)]

    if device == "cuda":
        # downgrade compute_type first, then model size, then CPU
        if (model_size, compute_type) != ("large-v2", "int8_float16"):
            chain.append(("large-v2", "cuda", "int8_float16"))
        if model_size != "medium":
            chain.append(("medium", "cuda", "int8_float16"))
        if model_size != "small":
            chain.append(("small", "cuda", "int8_float16"))
        chain.append(("small", "cpu", "int8_float32"))
    else:
        # CPU path: try a smaller model if available
        if model_size != "base":
            chain.append(("base", "cpu", "int8_float32"))

    # Collapse duplicates while preserving order
    dedup = []
    seen = set()
    for t in chain:
        if t not in seen:
            seen.add(t)
            dedup.append(t)
    return dedup

# ------------------------------ core transcription ------------------------------ #

def transcribe_audio(filepath, output_path="outputs/transcript.txt", model_size="large-v2"):
    """
    Transcribes an audio file using Faster-Whisper.
    Outputs a timestamped transcript to the specified file.

    Now with:
    - auto model selection by GPU VRAM
    - OOM-aware fallbacks that step down compute_type and/or model size
    """
    # Choose an initial config, then construct a fallback chain.
    initial_model, initial_device, initial_compute = _initial_asr_config(preferred_model=model_size)
    fallbacks = _asr_fallback_chain(initial_model, initial_device, initial_compute)

    last_error = None
    model = None
    segments = None
    info = None

    print(f"[ASR] CUDA available: {torch.cuda.is_available()} | VRAM: {_gpu_vram_gb():.1f} GiB")
    print(f"[ASR] Trying configs in order: {fallbacks}")

    for idx, (m, dev, ctype) in enumerate(fallbacks, start=1):
        try:
            print(f"[ASR] ({idx}/{len(fallbacks)}) Loading Whisper model '{m}' on {dev} with compute_type='{ctype}'...")
            model = WhisperModel(m, device=dev, compute_type=ctype)

            # Beam size 5 is your default; keep it. You can lower to 1 if you want more speed.
            segments, info = model.transcribe(filepath, beam_size=5)

            # Success
            print(f"Detected language: {info.language}")
            break
        except Exception as e:
            # Record error and try the next config
            last_error = e
            err_msg = str(e)
            oom_hint = "CUDA" in err_msg or "out of memory" in err_msg.lower()
            print(f"[ASR] Failed with '{m}'/{dev}/{ctype}: {e.__class__.__name__}: {err_msg}")
            if oom_hint:
                print("[ASR] Hint: CUDA OOM detected, falling back to a lighter config…")
            else:
                print("[ASR] Falling back to a different config…")
            model = None
            segments = None
            info = None
            # continue to next config

    if segments is None or info is None:
        # All attempts failed
        raise RuntimeError(f"ASR initialisation/transcription failed after {len(fallbacks)} attempts.") from last_error

    # Write results
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    full_text = ""
    with open(output_path, "w", encoding="utf-8") as f:
        for segment in segments:
            line = f"[{segment.start:.2f}s - {segment.end:.2f}s]  {segment.text}\n"
            print(line, end="")
            f.write(line)
            full_text += segment.text + " "
    print(f"Transcript saved to {output_path}")
    return full_text.strip()

# ------------------------------------ CLI ------------------------------------ #

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python transcribe.py <audio_or_video_file>")
        sys.exit(1)
    inputfile = sys.argv[1]

    # Determine input type
    if is_video_file(inputfile):
        print("Input detected as video file.")
        audiofile = "outputs/tmp_audio.wav"
        cleanup_audio = True
        try:
            extract_audio_from_video(inputfile, audiofile)
        except Exception as e:
            print(f"Audio extraction failed: {e}")
            sys.exit(1)
    elif is_audio_file(inputfile):
        print("Input detected as audio file.")
        audiofile = inputfile
        cleanup_audio = False
    else:
        print("Unsupported file type. Please provide an audio or video file.")
        sys.exit(1)

    # Run transcription
    try:
        # NOTE: still uses default output path to keep behavior consistent with your pipeline.
        # If you later add --output, just pass it through here.
        transcribe_audio(audiofile)
    except Exception as e:
        print(f"Transcription failed: {e}")
        sys.exit(1)
    finally:
        # Remove temporary audio file if created
        if cleanup_audio and os.path.exists(audiofile):
            try:
                os.remove(audiofile)
                print(f"Removed temporary audio file: {audiofile}")
            except Exception:
                pass
