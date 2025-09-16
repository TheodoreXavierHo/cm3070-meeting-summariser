# core/pipeline.py

# End-to-end CLI: ASR -> (OCR) -> Combine -> Summarise -> Actions -> metrics.json
# Usage:
#   python core/pipeline.py <media_file> [--skip-ocr] [--interval 5] [--lang en] [--cpu]
#   Optional: --output-dir outputs --frames-dir data/frames

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Ensure repository root is importable when executed as a script (python core/pipeline.py)
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.constants import VIDEO_EXTS

# Canonical output paths (match the app)
TRANSCRIPT = "outputs/transcript.txt"
SLIDES     = "outputs/slide_texts.txt"
COMBINED   = "outputs/combined_transcript.txt"
SUMMARY    = "outputs/summary.txt"
ACTIONS    = "outputs/action_items.txt"

def is_video(path: str) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTS

def run_step(desc: str, cmd: list, logf) -> float:
    """Run a step, stream logs to file/console, return elapsed seconds or raise."""
    print(f"[{desc}] -> {' '.join(cmd)}")
    logf.write(f"\n[{desc}] CMD: {' '.join(cmd)}\n")
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = round(time.time() - t0, 2)
    logf.write(proc.stdout or "")
    if proc.returncode != 0:
        logf.write(proc.stderr or "")
        raise RuntimeError(f"{desc} failed ({elapsed}s). See logs.")
    if proc.stderr:
        # keep stderr tail for later debugging
        logf.write("\n[stderr]\n" + (proc.stderr[-2000:]) + "\n")
    print(f"[{desc}] OK in {elapsed}s")
    return elapsed

def safe_read(p: str) -> str:
    try:
        with open(p, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""
    except Exception:
        return ""

def combine(transcript_path: str, slides_path: str, combined_path: str):
    t = safe_read(transcript_path)
    s = safe_read(slides_path) if slides_path and os.path.exists(slides_path) else ""
    if s and t:
        combined = f"--- Slide OCR Text ---\n{s}\n\n--- Transcript ---\n{t}"
    elif s:
        combined = f"--- Slide OCR Text ---\n{s}"
    else:
        combined = t
    with open(combined_path, "w", encoding="utf-8") as f:
        f.write(combined)

def ensure_dirs(output_dir: Path, frames_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    if frames_dir.exists():
        shutil.rmtree(frames_dir, ignore_errors=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

def main():
    ap = argparse.ArgumentParser(description="Run end-to-end meeting pipeline")
    ap.add_argument("media", help="Path to meeting audio/video file")
    ap.add_argument("--skip-ocr", action="store_true", help="Skip slide OCR (use transcript only)")
    ap.add_argument("--interval", type=int, default=5, help="OCR frame interval (sec)")
    ap.add_argument("--lang", default="en", help="OCR language code")
    ap.add_argument("--cpu", action="store_true", help="Force OCR on CPU")
    ap.add_argument("--output-dir", default="outputs", help="Directory for outputs")
    ap.add_argument("--frames-dir", default="data/frames", help="Directory for extracted frames")
    args = ap.parse_args()

    media = args.media
    if not os.path.exists(media):
        print(f"Input not found: {media}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    frames_dir = Path(args.frames_dir)
    ensure_dirs(output_dir, frames_dir)

    # absolute paths to match app expectations
    out_transcript = str(output_dir / Path(TRANSCRIPT).name)
    out_slides     = str(output_dir / Path(SLIDES).name)
    out_combined   = str(output_dir / Path(COMBINED).name)
    out_summary    = str(output_dir / Path(SUMMARY).name)
    out_actions    = str(output_dir / Path(ACTIONS).name)
    out_metrics    = str(output_dir / "metrics.json")
    out_log        = str(output_dir / "run.log")

    metrics = {}
    t0 = time.time()

    with open(out_log, "w", encoding="utf-8") as logf:
        try:
            # 1) ASR
            metrics["t_asr_s"] = run_step(
                "ASR",
                [sys.executable, "core/transcribe.py", media],
                logf,
            )

            # 2) OCR (video only & not skipped)
            do_ocr = is_video(media) and not args.skip_ocr
            if do_ocr:
                cmd = [
                    [sys.executable, "core/video_ocr.py", media,
                     "--output", out_slides,
                     "--frames_dir", str(frames_dir),
                     "--interval", str(max(1, min(15, args.interval))),
                     "--lang", args.lang],
                ]
                if args.cpu:
                    cmd[0].append("--cpu")
                metrics["t_ocr_s"] = run_step("OCR", cmd[0], logf)
            else:
                metrics["t_ocr_s"] = 0.0
                # ensure empty slide file removed to avoid confusion
                if os.path.exists(out_slides):
                    os.remove(out_slides)

            # 3) Combine
            t = time.time()
            combine(out_transcript, out_slides if do_ocr else "", out_combined)
            metrics["t_combine_s"] = round(time.time() - t, 2)

            # 4) Summarise
            metrics["t_summary_s"] = run_step(
                "Summary",
                [sys.executable, "core/summarise.py", out_combined],
                logf,
            )

            # 5) Action items
            metrics["t_actions_s"] = run_step(
                "Actions",
                [sys.executable, "core/extract_actions.py", out_combined],
                logf,
            )

            metrics["latency_total_s"] = round(time.time() - t0, 2)

        except Exception as e:
            metrics["error"] = str(e)
            print(f"[ERROR] {e}")

    # Write metrics (always)
    with open(out_metrics, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Echo summary to console
    if "error" in metrics:
        print(f"[DONE with errors] See {out_log} and {out_metrics}")
        sys.exit(1)
    else:
        print(f"[DONE] Metrics written to {out_metrics}")
        print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
