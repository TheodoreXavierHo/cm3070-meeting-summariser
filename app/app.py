# app/app.py

# Multimodal Meeting & Video Summariser UI (Upload → Preview → Process)
# Streamlit 1.45+ hardened, with sidebar expanders & rerun buttons moved.

import streamlit as st
import os, sys, re, time, hashlib, shutil, io
import subprocess
from typing import Dict, Tuple, List, Optional

st.set_page_config(
    page_title="Meeting & Video Summariser",
    layout="wide",
    initial_sidebar_state="collapsed"   # start with sidebar closed
)

SS = st.session_state
def _init_state():
    SS.setdefault("file_hash", None)
    SS.setdefault("upload_name", None)
    SS.setdefault("upload_ext", None)
    SS.setdefault("upload_bytes", None)
    SS.setdefault("paths", {})
    SS.setdefault("results", None)
    SS.setdefault("last_run_secs", 0.0)
    SS.setdefault("meta", {})
    SS.setdefault("is_video", False)
_init_state()

os.makedirs("data/samples", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("data/frames", exist_ok=True)

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".flv", ".webm"}
AUDIO_EXTS = {".wav", ".mp3", ".aac", ".ogg", ".flac", ".m4a", ".wma"}
MAX_PREVIEW_MB = 75

TRANSCRIPT = "outputs/transcript.txt"
SLIDES     = "outputs/slide_texts.txt"
COMBINED   = "outputs/combined_transcript.txt"
SUMMARY    = "outputs/summary.txt"
ACTIONS    = "outputs/action_items.txt"

def is_video(name: str) -> bool:
    return os.path.splitext(name)[1].lower() in VIDEO_EXTS

def file_md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()

def save_bytes_to(path: str, data: bytes):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)

def read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""

def run(cmd: List[str], env=None, desc="Running...") -> Tuple[bool, str]:
    st.info(desc)
    try:
        res = subprocess.run(cmd, text=True, capture_output=True, env=env)
        ok = (res.returncode == 0)
        logs = (res.stdout or "") + (res.stderr or "")
        if not ok:
            st.warning(f"{desc} failed.\n\n```\n{logs[-1200:]}\n```")
        return ok, logs
    except FileNotFoundError as e:
        st.warning(f"{desc} failed (command not found): {e}")
        return False, str(e)
    except Exception as e:
        st.warning(f"{desc} crashed: {e}")
        return False, str(e)

def clear_outputs():
    for p in [TRANSCRIPT, SLIDES, COMBINED, SUMMARY, ACTIONS, "outputs/metrics.json"]:
        try:
            if os.path.exists(p): os.remove(p)
        except Exception:
            pass
    try:
        if os.path.isdir("data/frames"): shutil.rmtree("data/frames")
    finally:
        os.makedirs("data/frames", exist_ok=True)

def combine(transcript_path: str, slides_path: Optional[str], combined_path: str) -> str:
    t = read_text(transcript_path)
    s = read_text(slides_path) if slides_path and os.path.exists(slides_path) else ""
    if s and t:
        combined = f"--- Slide OCR Text ---\n{s}\n\n--- Transcript ---\n{t}"
    elif s:
        combined = f"--- Slide OCR Text ---\n{s}"
    else:
        combined = t
    with open(combined_path, "w", encoding="utf-8") as f:
        f.write(combined)
    return combined

def probe_video_meta(path: str) -> Dict:
    meta = {}
    try:
        import cv2
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            cap.release(); return meta
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()
        dur = frames / fps if fps > 0 else 0.0
        meta.update({"fps": round(float(fps), 2), "frames": int(frames),
                     "duration_s": round(float(dur), 1), "resolution": f"{w}×{h}"})
    except Exception:
        pass
    return meta

def human_size(n: int) -> str:
    size = float(n)
    for unit in ["B","KB","MB","GB","TB","PB"]:
        if size < 1024: return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} EB"

# ---------------- Sidebar (expanders) ----------------
with st.sidebar:
    # SETTINGS (closed by default)
    with st.expander("Settings", expanded=False):
        ocr_interval = int(st.number_input("OCR frame interval (sec)", min_value=1, max_value=15, value=5, step=1, key="ocr_interval"))
        ocr_lang = st.text_input("OCR language code", value="en", key="ocr_lang").strip() or "en"
        ocr_cpu = st.checkbox("Force OCR on CPU", value=False, key="ocr_cpu")
        skip_ocr = st.checkbox("Skip OCR for videos (use transcript only)", value=False, key="skip_ocr")

        st.markdown("---")
        quant = st.selectbox("Model quantisation", ["4bit","8bit","fp16","auto"], index=0, key="quant")
        repetition_penalty = float(st.slider("Repetition penalty", 1.00, 1.25, 1.15, 0.01, key="rep_pen"))
        no_repeat_ngram = int(st.slider("No-repeat n-gram size", 0, 12, 8, 1, key="no_rep_ng"))

        st.caption("These apply on the next **Process file** or partial re-run.")

    # RERUN / MAINTENANCE (also closed by default)
    with st.expander("Re-run / Maintenance", expanded=False):
        st.caption("Partial re-runs use the cached combined transcript (no ASR/OCR).")
        rerun_sum_btn = st.button("↻ Re-run summary only", use_container_width=True, disabled=SS.results is None, key="btn_rerun_sum")
        rerun_act_btn = st.button("↻ Re-run action items only", use_container_width=True, disabled=SS.results is None, key="btn_rerun_act")
        clear_btn     = st.button("🗑️ Clear results", use_container_width=True, disabled=(SS.results is None and SS.paths == {}), key="btn_clear")

def _env_base():
    env = os.environ.copy()
    env["FYP_QUANT"] = st.session_state.get("quant", "4bit")
    env["FYP_REPETITION_PENALTY"] = f'{st.session_state.get("rep_pen", 1.15):.3f}'
    env["FYP_NO_REPEAT_NGRAM"] = str(st.session_state.get("no_rep_ng", 8))
    return env

# ---------------- Upload & preview ----------------
st.title("📑 Multimodal Meeting & Video Summariser")

uploaded = st.file_uploader(
    "Upload meeting audio or video",
    type=sorted({e[1:] for e in (VIDEO_EXTS | AUDIO_EXTS)}),
    accept_multiple_files=False
)

if uploaded is not None:
    uploaded.seek(0); data = uploaded.read(); uploaded.seek(0)
    if not data:
        st.warning("Uploaded file appears to be empty."); st.stop()

    cur_hash = file_md5(data)
    ext = os.path.splitext(uploaded.name)[1].lower()
    disk_path = f"data/samples/input{ext}"

    if SS.file_hash != cur_hash:
        SS.file_hash = cur_hash
        SS.upload_name = uploaded.name
        SS.upload_ext = ext
        SS.results = None
        SS.paths = {}
        SS.last_run_secs = 0.0
        SS.meta = {}
        SS.is_video = is_video(uploaded.name)

        save_bytes_to(disk_path, data)
        SS.meta["size"] = human_size(len(data))
        if SS.is_video:
            SS.meta.update(probe_video_meta(disk_path))

        SS.upload_bytes = data if len(data) <= MAX_PREVIEW_MB * 1024 * 1024 else None

    st.success(f"File selected: {SS.upload_name}  •  Size: {SS.meta.get('size','?')}")
    if SS.is_video:
        if SS.meta:
            st.caption(f"Video meta — {SS.meta.get('resolution','?')} @ {SS.meta.get('fps','?')} fps • Duration: {SS.meta.get('duration_s','?')} s")
        if SS.upload_bytes is not None:
            st.video(io.BytesIO(SS.upload_bytes), format="video/mp4")
        else:
            st.info(f"Preview disabled for large files (> {MAX_PREVIEW_MB} MB).")
    else:
        if SS.upload_bytes is not None:
            st.audio(io.BytesIO(SS.upload_bytes))
        else:
            st.info(f"Preview disabled for large files (> {MAX_PREVIEW_MB} MB).")

    # Main action row: only Process file lives here now
    process_btn = st.button("▶️ Process file", type="primary", key="btn_process")

else:
    st.info("Upload a file to preview. Processing starts only when you click **Process file**.")
    st.stop()

# Clear results handling (from sidebar)
if 'clear_btn' in locals() and clear_btn:
    SS.results = None
    SS.paths = {}
    SS.last_run_secs = 0.0
    clear_outputs()
    st.success("Results cleared.")
    st.stop()

# ---------------- Pipeline runners ----------------
def _process_pipeline() -> bool:
    start = time.time()
    clear_outputs()

    input_path = f"data/samples/input{SS.upload_ext}"
    if not os.path.exists(input_path):
        if SS.upload_bytes is not None:
            save_bytes_to(input_path, SS.upload_bytes)
        else:
            st.warning("Input file missing on disk and too large to re-buffer. Please re-upload.")
            return False

    prog = st.progress(0.0)
    status = st.empty()

    status.info("Step 1/5 — Transcribing (Whisper)...")
    ok, _ = run([sys.executable, "core/transcribe.py", input_path], env=_env_base(), desc="Transcription")
    if not ok: status.empty(); prog.empty(); return False
    prog.progress(0.2)

    slides_exist = False
    if SS.is_video and not st.session_state.get("skip_ocr", False):
        status.info("Step 2/5 — Extracting slides (OCR)...")
        interval = int(max(1, min(15, st.session_state.get("ocr_interval", 5))))
        lang = st.session_state.get("ocr_lang", "en"); lang = lang if 1 <= len(lang) <= 8 else "en"
        cmd = [sys.executable, "core/video_ocr.py", input_path,
               "--output", SLIDES, "--frames_dir", "data/frames",
               "--interval", str(interval), "--lang", lang]
        if st.session_state.get("ocr_cpu", False):
            cmd.append("--cpu")
        ok, _ = run(cmd, env=_env_base(), desc="Slide OCR")
        if not ok: status.empty(); prog.empty(); return False
        slides_exist = os.path.exists(SLIDES) and os.path.getsize(SLIDES) > 0
    else:
        status.info("Step 2/5 — OCR skipped or not required.")
    prog.progress(0.4)

    status.info("Step 3/5 — Combining transcript and slides...")
    combine(TRANSCRIPT, SLIDES if slides_exist else "", COMBINED)
    prog.progress(0.6)

    status.info("Step 4/5 — Summarising (LLM)...")
    ok, _ = run([sys.executable, "core/summarise.py", COMBINED], env=_env_base(), desc="Summarisation")
    if not ok: status.empty(); prog.empty(); return False
    prog.progress(0.8)

    status.info("Step 5/5 — Extracting action items...")
    ok, _ = run([sys.executable, "core/extract_actions.py", COMBINED], env=_env_base(), desc="Action item extraction")
    if not ok: status.empty(); prog.empty(); return False
    prog.progress(1.0); status.success("Processing complete."); time.sleep(0.2)
    status.empty(); prog.empty()

    SS.results = {
        "slides": read_text(SLIDES),
        "transcript": read_text(TRANSCRIPT),
        "combined": read_text(COMBINED),
        "summary": read_text(SUMMARY),
        "actions": read_text(ACTIONS),
    }
    SS.paths = {"slides": SLIDES, "transcript": TRANSCRIPT, "combined": COMBINED,
                "summary": SUMMARY, "actions": ACTIONS}
    SS.last_run_secs = time.time() - start
    return True

def _rerun_summary_only() -> bool:
    if not os.path.exists(COMBINED):
        st.warning("Missing combined transcript. Please run the full pipeline once."); return False
    ok, _ = run([sys.executable, "core/summarise.py", COMBINED], env=_env_base(), desc="Summarisation (partial)")
    if ok:
        SS.results = SS.results or {}
        SS.results["summary"] = read_text(SUMMARY)
    return ok

def _rerun_actions_only() -> bool:
    if not os.path.exists(COMBINED):
        st.warning("Missing combined transcript. Please run the full pipeline once."); return False
    ok, _ = run([sys.executable, "core/extract_actions.py", COMBINED], env=_env_base(), desc="Action item extraction (partial)")
    if ok:
        SS.results = SS.results or {}
        SS.results["actions"] = read_text(ACTIONS)
    return ok

# Button triggers
if process_btn:
    _process_pipeline()
if 'rerun_sum_btn' in locals() and rerun_sum_btn and SS.results is not None:
    _rerun_summary_only()
if 'rerun_act_btn' in locals() and rerun_act_btn and SS.results is not None:
    _rerun_actions_only()

if not SS.results:
    st.stop()

# ---------------- Render helpers & tabs ----------------
def render_slides(slide_text: str):
    titles = re.findall(r'^--- Slide (\d+) ---$', slide_text, flags=re.MULTILINE)
    blocks = re.split(r'^--- Slide \d+ ---$', slide_text, flags=re.MULTILINE)
    slides = []
    for i in range(len(titles)):
        try: num = int(titles[i])
        except Exception: num = i + 1
        content = (blocks[i + 1] if i + 1 < len(blocks) else "").strip()
        slides.append((num, content))
    for num, content in slides:
        with st.expander(f"Slide {num}"):
            st.markdown(content.replace("\n", "  \n") if content else "_(empty)_")

def parse_transcript_segments(text: str) -> List[Tuple[str,str]]:
    return re.findall(r'^\[(.*?)\]\s+(.*)$', text or "", flags=re.MULTILINE)

def render_transcript(text: str):
    segs = parse_transcript_segments(text or "")
    if not segs:
        st.text_area("Transcript", text or "", height=320); return
    st.markdown("""
        <style>
        .tswrap{background:#1f1f1f;padding:12px;border-radius:12px;max-height:420px;overflow-y:auto;}
        .tsr{display:flex;gap:.75rem;margin:.35rem 0}
        .tst{min-width:110px;color:#7fb0ff;font-family:monospace}
        .tsu{flex:1;color:#ddd}
        </style>
    """, unsafe_allow_html=True)
    st.markdown('<div class="tswrap">', unsafe_allow_html=True)
    for ts, utt in segs:
        st.markdown(f'<div class="tsr"><div class="tst">{ts}</div><div class="tsu">{utt}</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def parse_action_items(md: str) -> List[Dict[str,str]]:
    items: List[Dict[str,str]] = []
    if not md: return items
    blocks = re.split(r'\n\s*\d+\.\s+', "\n" + md.strip())
    for block in blocks:
        block = block.strip()
        if not block: continue
        entry = {"Task":"", "Owner":"", "Deadline":""}
        m_task = re.search(r'\*\*(.+?)\*\*', block)
        entry["Task"] = (m_task.group(1).strip() if m_task else block.splitlines()[0].strip())
        m_owner = re.search(r'\*\*Owner:\*\*\s*([^\n|—-]+)', block, flags=re.IGNORECASE) or \
                  re.search(r'Owner\s*:\s*([^\n|—-]+)', block, flags=re.IGNORECASE)
        m_dead  = re.search(r'\*\*Deadline:\*\*\s*([^\n|—-]+)', block, flags=re.IGNORECASE) or \
                  re.search(r'Deadline\s*:\s*([^\n|—-]+)', block, flags=re.IGNORECASE)
        entry["Owner"] = (m_owner.group(1).strip() if m_owner else "")
        entry["Deadline"] = (m_dead.group(1).strip() if m_dead else "")
        if entry["Task"]: items.append(entry)
    return items

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
        st.table(items)   # native, no pandas
    else:
        st.markdown(SS.results.get("actions") or "_No action items_")
    st.download_button("Download action_items.txt", SS.results.get("actions") or "", "action_items.txt")
