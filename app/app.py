# app/app.py
# -----------------------------------------------------------------------------
# Multimodal Meeting & Video Summariser (Streamlit UI)
# - Upload → Preview → Process (ASR → OCR → Combine → Summarise → Action Items)
# -----------------------------------------------------------------------------

from __future__ import annotations

import io
import os
import re
import sys
import time
import json
import shutil
import hashlib
import secrets
import subprocess
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import streamlit as st

from core.constants import AUDIO_EXTS, VIDEO_EXTS

# ------------------------------ page config ----------------------------------

st.set_page_config(
    page_title="Meeting & Video Summariser",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ------------------------------ constants ------------------------------------

# Core scripts expect to read/write these fixed paths. We mirror them per-run.
OUTPUTS_DIR = Path("outputs")
FRAMES_DIR = Path("data/frames")
SAMPLES_DIR = Path("data/samples")

TRANSCRIPT_FIXED = OUTPUTS_DIR / "transcript.txt"
SLIDES_FIXED = OUTPUTS_DIR / "slide_texts.txt"
COMBINED_FIXED = OUTPUTS_DIR / "combined_transcript.txt"  # not used by core, we write per-run
SUMMARY_FIXED = OUTPUTS_DIR / "summary.txt"
ACTIONS_FIXED = OUTPUTS_DIR / "action_items.txt"

MAX_PREVIEW_MB = 75

# Allow-list a few OCR language tags (extend if you enable EasyOCR langs)
OCR_LANG_ALLOW = {"en", "fr", "de", "es", "it", "pt", "nl", "ja", "ko", "zh"}


# ------------------------------ helpers --------------------------------------

SS = st.session_state


def _ensure_dirs() -> None:
    """Create base folders if missing."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)


def _init_state() -> None:
    """Initialise once-per-session state keys."""
    SS.setdefault("file_sha256", None)
    SS.setdefault("upload_name", None)
    SS.setdefault("upload_ext", None)
    SS.setdefault("upload_bytes", None)
    SS.setdefault("is_video", False)

    # run-scoped state
    SS.setdefault("run_id", None)
    SS.setdefault("run_dir", None)        # outputs/<run_id>
    SS.setdefault("paths", {})            # resolved per-run artefact paths
    SS.setdefault("results", None)        # cached text payloads for tabs
    SS.setdefault("last_run_secs", 0.0)
    SS.setdefault("meta", {})

    # simple state machine: idle | running
    SS.setdefault("phase", "idle")


_ensure_dirs()
_init_state()


def is_video(name: str) -> bool:
    return Path(name).suffix.lower() in VIDEO_EXTS


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def human_size(n: int) -> str:
    size = float(n)
    for unit in ["B", "KB", "MB", "GB", "TB", "PB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} EB"


def save_bytes_to(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(data)


def read_text(path: Path) -> str:
    """Robust text read with UTF-8 + replacement for odd bytes."""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def copy_if_exists(src: Path, dst: Path) -> None:
    """Copy file if present; create parent of dst."""
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def clear_fixed_outputs() -> None:
    """Clear global outputs (used by core scripts) and frames cache."""
    for p in [TRANSCRIPT_FIXED, SLIDES_FIXED, COMBINED_FIXED, SUMMARY_FIXED, ACTIONS_FIXED]:
        try:
            if p.exists():
                p.unlink()
        except Exception:
            pass
    # Reset frames directory
    try:
        if FRAMES_DIR.is_dir():
            shutil.rmtree(FRAMES_DIR)
    finally:
        FRAMES_DIR.mkdir(parents=True, exist_ok=True)


def new_run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(4)


def build_run_paths(run_dir: Path) -> Dict[str, Path]:
    """Return per-run artefact paths (we snapshot fixed outputs into here)."""
    return {
        "input": run_dir / "input",  # we'll append the proper extension later
        "transcript": run_dir / "transcript.txt",
        "slides": run_dir / "slide_texts.txt",
        "combined": run_dir / "combined_transcript.txt",
        "summary": run_dir / "summary.txt",
        "actions": run_dir / "action_items.txt",
        "params": run_dir / "params.json",
        "log": run_dir / "run.log",
    }


def write_params(run_paths: Dict[str, Path], params: Dict) -> None:
    try:
        run_paths["params"].write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def run_subprocess(
    cmd: List[str],
    *,
    env: Optional[Dict[str, str]] = None,
    desc: str = "Running…",
    timeout: Optional[int] = None,
) -> Tuple[bool, str, int]:
    """Run a subprocess with log capture and better error surfacing."""
    st.info(desc)
    try:
        started = time.time()
        res = subprocess.run(cmd, text=True, capture_output=True, env=env, timeout=timeout)
        duration = time.time() - started
        logs = (res.stdout or "") + (res.stderr or "")
        ok = res.returncode == 0
        if not ok:
            st.error(f"{desc} failed (exit {res.returncode}, {duration:.1f}s). See log below.")
            st.code(logs[-4000:])
            st.download_button("Download full log", logs, file_name="step.log")
        return ok, logs, res.returncode
    except subprocess.TimeoutExpired as e:
        msg = f"{desc} timed out after {timeout}s"
        st.error(msg)
        return False, msg, -1
    except FileNotFoundError as e:
        msg = f"{desc} failed (command not found): {e}"
        st.error(msg)
        return False, msg, -2
    except Exception as e:
        msg = f"{desc} crashed: {e}"
        st.error(msg)
        return False, msg, -3


def probe_video_meta(path: Path) -> Dict:
    """Best-effort probe of duration, fps, resolution via OpenCV."""
    meta: Dict[str, object] = {}
    try:
        import cv2

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            cap.release()
            return meta
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()
        dur = frames / fps if fps > 0 else 0.0
        meta.update(
            {"fps": round(float(fps), 2), "frames": int(frames), "duration_s": round(float(dur), 1), "resolution": f"{w}×{h}"}
        )
    except Exception:
        pass
    return meta


def combine_inputs(transcript_path: Path, slides_path: Optional[Path], combined_path: Path) -> str:
    """Write the combined input (slide OCR + transcript) to per-run combined file."""
    t = read_text(transcript_path)
    s = read_text(slides_path) if slides_path and slides_path.exists() else ""
    if s and t:
        combined = f"--- Slide OCR Text ---\n{s}\n\n--- Transcript ---\n{t}"
    elif s:
        combined = f"--- Slide OCR Text ---\n{s}"
    else:
        combined = t
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    combined_path.write_text(combined, encoding="utf-8")
    return combined


def env_from_sidebar() -> Dict[str, str]:
    """Build environment variables consumed by your core scripts."""
    env = os.environ.copy()
    env["FYP_QUANT"] = st.session_state.get("quant", "4bit")
    env["FYP_REPETITION_PENALTY"] = f'{st.session_state.get("rep_pen", 1.15):.3f}'
    env["FYP_NO_REPEAT_NGRAM"] = str(st.session_state.get("no_rep_ng", 8))
    return env


# ------------------------------ sidebar --------------------------------------

with st.sidebar:
    with st.expander("Settings", expanded=False):
        # OCR settings
        ocr_interval = int(
            st.number_input("OCR frame interval (sec)", min_value=1, max_value=15, value=5, step=1, key="ocr_interval")
        )
        raw_lang = st.text_input("OCR language code", value="en", key="ocr_lang").strip().lower() or "en"
        # Validate OCR language against allow-list; fall back to 'en'
        ocr_lang = raw_lang if raw_lang in OCR_LANG_ALLOW else "en"

        ocr_cpu = st.checkbox("Force OCR on CPU", value=False, key="ocr_cpu")
        skip_ocr = st.checkbox("Skip OCR for videos (use transcript only)", value=False, key="skip_ocr")

        st.markdown("---")
        # LLM generation settings
        quant = st.selectbox("Model quantisation", ["4bit", "8bit", "fp16", "auto"], index=0, key="quant")
        repetition_penalty = float(st.slider("Repetition penalty", 1.00, 1.25, 1.15, 0.01, key="rep_pen"))
        no_repeat_ngram = int(st.slider("No-repeat n-gram size", 0, 12, 8, 1, key="no_rep_ng"))

        st.caption("These apply on the next **Process file** or partial re-run.")

    with st.expander("Re-run / Maintenance", expanded=False):
        st.caption("Partial re-runs use the cached combined transcript (no ASR/OCR).")
        rerun_sum_btn = st.button(
            "↻ Re-run summary only", use_container_width=True, disabled=SS.results is None or SS.phase != "idle", key="btn_rerun_sum"
        )
        rerun_act_btn = st.button(
            "↻ Re-run action items only", use_container_width=True, disabled=SS.results is None or SS.phase != "idle", key="btn_rerun_act"
        )
        clear_btn = st.button(
            "🗑️ Clear results", use_container_width=True, disabled=(SS.results is None and not SS.run_dir) or SS.phase != "idle", key="btn_clear"
        )

# ------------------------------ upload & preview -----------------------------

st.title("📑 Multimodal Meeting & Video Summariser")

uploaded = st.file_uploader(
    "Upload meeting audio or video",
    type=sorted({e[1:] for e in (VIDEO_EXTS | AUDIO_EXTS)}),
    accept_multiple_files=False,
)

if uploaded is None:
    st.info("Upload a file to preview. Processing starts only when you click **Process file**.")
    st.stop()

# Read bytes safely and compute content hash
uploaded.seek(0)
data = uploaded.read()
uploaded.seek(0)
if not data:
    st.warning("Uploaded file appears to be empty.")
    st.stop()

cur_sha = sha256_of(data)
ext = Path(uploaded.name).suffix.lower()

# If a *new* file is selected, reset run state and snapshot the upload
if SS.file_sha256 != cur_sha:
    SS.file_sha256 = cur_sha
    SS.upload_name = uploaded.name
    SS.upload_ext = ext
    SS.is_video = is_video(uploaded.name)

    # Create a new run folder and compute its per-run paths
    SS.run_id = new_run_id()
    SS.run_dir = (OUTPUTS_DIR / SS.run_id).resolve()
    SS.paths = build_run_paths(SS.run_dir)

    # Persist input in the run folder AND at the fixed path expected by core scripts
    SS.paths["input"] = SS.paths["input"].with_suffix(ext)
    save_bytes_to(SS.paths["input"], data)
    save_bytes_to(SAMPLES_DIR / f"input{ext}", data)  # core/transcribe.py reads this

    # Reset results & meta
    SS.results = None
    SS.last_run_secs = 0.0
    SS.meta = {"size": human_size(len(data))}
    if SS.is_video:
        SS.meta.update(probe_video_meta(SS.paths["input"]))

    # Store params that produced this run (handy for provenance)
    write_params(
        SS.paths,
        {
            "ocr_interval": ocr_interval,
            "ocr_lang": ocr_lang,
            "ocr_cpu": ocr_cpu,
            "skip_ocr": skip_ocr,
            "quant": quant,
            "rep_pen": repetition_penalty,
            "no_rep_ng": no_repeat_ngram,
        },
    )

# Preview
st.success(f"File selected: {SS.upload_name}  •  Size: {SS.meta.get('size','?')}")
if SS.is_video:
    if SS.meta:
        st.caption(
            f"Video meta — {SS.meta.get('resolution','?')} @ {SS.meta.get('fps','?')} fps • Duration: {SS.meta.get('duration_s','?')} s"
        )
    if len(data) <= MAX_PREVIEW_MB * 1024 * 1024:
        st.video(io.BytesIO(data), format="video/mp4")
    else:
        st.info(f"Preview disabled for large files (> {MAX_PREVIEW_MB} MB).")
else:
    if len(data) <= MAX_PREVIEW_MB * 1024 * 1024:
        st.audio(io.BytesIO(data))
    else:
        st.info(f"Preview disabled for large files (> {MAX_PREVIEW_MB} MB).")

# Main action row
process_btn = st.button("▶️ Process file", type="primary", disabled=SS.phase != "idle", key="btn_process")


# ------------------------------ actions: clear / partial reruns --------------

if clear_btn:
    # Clear only this run's snapshots + fixed outputs (safer than global nuke)
    try:
        if SS.run_dir and Path(SS.run_dir).exists():
            shutil.rmtree(SS.run_dir)
    except Exception:
        pass
    clear_fixed_outputs()
    SS.results = None
    SS.paths = {}
    SS.last_run_secs = 0.0
    SS.run_id = None
    SS.run_dir = None
    st.success("Results cleared.")
    st.stop()


def _snapshot_fixed_into_run() -> None:
    """Copy global outputs (written by core scripts) into the current run dir."""
    copy_if_exists(TRANSCRIPT_FIXED, SS.paths["transcript"])
    copy_if_exists(SLIDES_FIXED, SS.paths["slides"])
    copy_if_exists(SUMMARY_FIXED, SS.paths["summary"])
    copy_if_exists(ACTIONS_FIXED, SS.paths["actions"])


def _process_pipeline() -> bool:
    """Full pipeline: ASR → (OCR) → Combine → Summarise → Action Items."""
    SS.phase = "running"
    start = time.time()
    clear_fixed_outputs()

    # Ensure the input the cores expect exists (in case of browser refresh)
    core_input = SAMPLES_DIR / f"input{SS.upload_ext}"
    if not core_input.exists():
        save_bytes_to(core_input, SS.paths["input"].read_bytes())

    prog = st.progress(0.0)
    status = st.empty()

    # 1) ASR
    status.info("Step 1/5 — Transcribing (Whisper)…")
    ok, _, _ = run_subprocess(
        [sys.executable, "core/transcribe.py", str(core_input)],
        env=env_from_sidebar(),
        desc="Transcription",
    )
    if not ok:
        SS.phase = "idle"
        status.empty()
        prog.empty()
        return False
    prog.progress(0.2)
    _snapshot_fixed_into_run()

    # 2) OCR (video only, unless skipped)
    slides_exist = False
    if SS.is_video and not skip_ocr:
        status.info("Step 2/5 — Extracting slides (OCR)…")
        cmd = [
            sys.executable,
            "core/video_ocr.py",
            str(core_input),
            "--output",
            str(SLIDES_FIXED),
            "--frames_dir",
            str(FRAMES_DIR),
            "--interval",
            str(int(max(1, min(15, ocr_interval)))),
            "--lang",
            ocr_lang,
        ]
        if ocr_cpu:
            cmd.append("--cpu")
        ok, _, _ = run_subprocess(cmd, env=env_from_sidebar(), desc="Slide OCR")
        if not ok:
            SS.phase = "idle"
            status.empty()
            prog.empty()
            return False
        slides_exist = SLIDES_FIXED.exists() and SLIDES_FIXED.stat().st_size > 0
        _snapshot_fixed_into_run()
    else:
        status.info("Step 2/5 — OCR skipped or not required.")
    prog.progress(0.4)

    # 3) Combine (always per-run file; no need to write global combined)
    status.info("Step 3/5 — Combining transcript and slides…")
    combine_inputs(SS.paths["transcript"], SS.paths["slides"] if slides_exist else None, SS.paths["combined"])
    prog.progress(0.6)

    # 4) Summarise (reads combined from per-run; core writes global summary)
    status.info("Step 4/5 — Summarising (LLM)…")
    ok, _, _ = run_subprocess(
        [sys.executable, "core/summarise.py", str(SS.paths["combined"])],
        env=env_from_sidebar(),
        desc="Summarisation",
    )
    if not ok:
        SS.phase = "idle"
        status.empty()
        prog.empty()
        return False
    _snapshot_fixed_into_run()
    prog.progress(0.8)

    # 5) Action items (reads combined from per-run; core writes global actions)
    status.info("Step 5/5 — Extracting action items…")
    ok, _, _ = run_subprocess(
        [sys.executable, "core/extract_actions.py", str(SS.paths["combined"])],
        env=env_from_sidebar(),
        desc="Action item extraction",
    )
    if not ok:
        SS.phase = "idle"
        status.empty()
        prog.empty()
        return False
    _snapshot_fixed_into_run()
    prog.progress(1.0)

    # Load results into session
    SS.results = {
        "slides": read_text(SS.paths["slides"]),
        "transcript": read_text(SS.paths["transcript"]),
        "combined": read_text(SS.paths["combined"]),
        "summary": read_text(SS.paths["summary"]),
        "actions": read_text(SS.paths["actions"]),
    }
    SS.last_run_secs = time.time() - start

    status.success("Processing complete.")
    time.sleep(0.2)
    status.empty()
    prog.empty()
    SS.phase = "idle"
    return True


def _rerun_summary_only() -> bool:
    """Partial re-run: only summary (LLM)."""
    if not SS.paths or not SS.paths.get("combined") or not SS.paths["combined"].exists():
        st.warning("Missing combined transcript. Please run the full pipeline once.")
        return False
    ok, _, _ = run_subprocess(
        [sys.executable, "core/summarise.py", str(SS.paths["combined"])],
        env=env_from_sidebar(),
        desc="Summarisation (partial)",
    )
    if ok:
        copy_if_exists(SUMMARY_FIXED, SS.paths["summary"])
        SS.results = SS.results or {}
        SS.results["summary"] = read_text(SS.paths["summary"])
    return ok


def _rerun_actions_only() -> bool:
    """Partial re-run: only action item extraction."""
    if not SS.paths or not SS.paths.get("combined") or not SS.paths["combined"].exists():
        st.warning("Missing combined transcript. Please run the full pipeline once.")
        return False
    ok, _, _ = run_subprocess(
        [sys.executable, "core/extract_actions.py", str(SS.paths["combined"])],
        env=env_from_sidebar(),
        desc="Action item extraction (partial)",
    )
    if ok:
        copy_if_exists(ACTIONS_FIXED, SS.paths["actions"])
        SS.results = SS.results or {}
        SS.results["actions"] = read_text(SS.paths["actions"])
    return ok


# Trigger buttons (guarded by state)
if process_btn:
    _process_pipeline()
if rerun_sum_btn and SS.results is not None and SS.phase == "idle":
    _rerun_summary_only()
if rerun_act_btn and SS.results is not None and SS.phase == "idle":
    _rerun_actions_only()

# If no results yet, stop rendering tabs
if not SS.results:
    st.stop()

# ------------------------------ rendering helpers ----------------------------

def parse_transcript_segments(text: str) -> List[Tuple[str, str]]:
    """Parse lines like: [hh:mm:ss] utterance..."""
    return re.findall(r"^\[(.*?)\]\s+(.*)$", text or "", flags=re.MULTILINE)


def render_transcript(text: str) -> None:
    segs = parse_transcript_segments(text or "")
    if not segs:
        st.text_area("Transcript", text or "", height=320)
        return
    st.markdown(
        """
        <style>
        .tswrap{background:#1f1f1f;padding:12px;border-radius:12px;max-height:420px;overflow-y:auto;}
        .tsr{display:flex;gap:.75rem;margin:.35rem 0}
        .tst{min-width:110px;color:#7fb0ff;font-family:monospace}
        .tsu{flex:1;color:#ddd}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="tswrap">', unsafe_allow_html=True)
    for ts, utt in segs:
        st.markdown(f'<div class="tsr"><div class="tst">{ts}</div><div class="tsu">{utt}</div></div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_slides(slide_text: str) -> None:
    """Render blocks marked as:
       --- Slide N ---
       content...
    """
    titles = re.findall(r"^--- Slide (\d+) ---$", slide_text or "", flags=re.MULTILINE)
    blocks = re.split(r"^--- Slide \d+ ---$", slide_text or "", flags=re.MULTILINE)
    slides: List[Tuple[int, str]] = []
    for i in range(len(titles)):
        try:
            num = int(titles[i])
        except Exception:
            num = i + 1
        content = (blocks[i + 1] if i + 1 < len(blocks) else "").strip()
        slides.append((num, content))
    for num, content in slides:
        with st.expander(f"Slide {num}"):
            st.markdown(content.replace("\n", "  \n") if content else "_(empty)_")


def parse_action_items(md: str) -> List[Dict[str, str]]:
    """Parse simple markdown list like:
       1. **Task** …\nOwner: X\nDeadline: Y
       (Tolerant to bold/no-bold labels and different dash types.)
    """
    items: List[Dict[str, str]] = []
    if not md:
        return items
    blocks = re.split(r"\n\s*\d+\.\s+", "\n" + md.strip())
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        entry = {"Task": "", "Owner": "", "Deadline": ""}
        m_task = re.search(r"\*\*(.+?)\*\*", block)
        entry["Task"] = (m_task.group(1).strip() if m_task else block.splitlines()[0].strip())
        m_owner = re.search(r"\*\*Owner:\*\*\s*([^\n|—-]+)", block, flags=re.IGNORECASE) or re.search(
            r"Owner\s*:\s*([^\n|—-]+)", block, flags=re.IGNORECASE
        )
        m_dead = re.search(r"\*\*Deadline:\*\*\s*([^\n|—-]+)", block, flags=re.IGNORECASE) or re.search(
            r"Deadline\s*:\s*([^\n|—-]+)", block, flags=re.IGNORECASE
        )
        entry["Owner"] = (m_owner.group(1).strip() if m_owner else "")
        entry["Deadline"] = (m_dead.group(1).strip() if m_dead else "")
        if entry["Task"]:
            items.append(entry)
    return items


# ------------------------------ tabs -----------------------------------------

st.success(f"⏱️ Total processing time: {int(SS.last_run_secs//60)} min {int(SS.last_run_secs%60)} sec")

tabs = st.tabs(["Slides (OCR)", "Transcript", "Combined", "Summary", "Action items"])

with tabs[0]:
    st.markdown("#### Slides (OCR)")
    if (SS.results.get("slides") or "").strip():
        render_slides(SS.results["slides"])
        st.download_button("Download slide_texts.txt", SS.results["slides"], "slide_texts.txt")
    else:
        st.info("No slides (audio file or OCR skipped).")

with tabs[1]:
    st.markdown("#### Transcript")
    render_transcript(SS.results.get("transcript") or "")
    st.download_button("Download transcript.txt", SS.results.get("transcript") or "", "transcript.txt")

with tabs[2]:
    st.markdown("#### Combined input")
    st.text_area("Combined transcript (slides + speech)", SS.results.get("combined") or "", height=320)
    st.download_button("Download combined_transcript.txt", SS.results.get("combined") or "", "combined_transcript.txt")

with tabs[3]:
    st.markdown("#### Meeting summary")
    st.markdown(SS.results.get("summary") or "_No summary_")
    st.download_button("Download summary.txt", SS.results.get("summary") or "", "summary.txt")

with tabs[4]:
    st.markdown("#### Action items")
    items = parse_action_items(SS.results.get("actions") or "")
    if items:
        # dataframe allows sorting/filtering; native table is static
        st.dataframe(items, use_container_width=True)
    else:
        st.markdown(SS.results.get("actions") or "_No action items_")
    st.download_button("Download action_items.txt", SS.results.get("actions") or "", "action_items.txt")
