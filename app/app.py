# core/app.py

import streamlit as st
import os
import subprocess
import sys
import shutil
import re
import time

# ---------- Utility functions ----------

def is_video_file(filename):
    return os.path.splitext(filename)[1].lower() in {'.mp4', '.mov', '.mkv', '.avi', '.flv', '.webm'}

def is_audio_file(filename):
    return os.path.splitext(filename)[1].lower() in {'.wav', '.mp3', '.aac', '.ogg', '.flac', '.m4a', '.wma'}

def run_subprocess(cmd, desc="Running command..."):
    """Runs a subprocess, displays warnings but only blocks if output is missing."""
    st.info(desc)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        st.warning(f"Warning during: {desc}\n{result.stderr}")
        return False
    return True

def safe_read(filepath):
    """Reads a text file, returns empty string if not found or path is None/empty."""
    if not filepath:
        return ""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""

def save_uploaded_file(uploaded_file, save_path):
    with open(save_path, "wb") as f:
        f.write(uploaded_file.read())

def combine_transcript_and_slides(transcript_path, slides_path, combined_path):
    transcript = safe_read(transcript_path)
    slides = safe_read(slides_path)
    if slides and transcript:
        combined = f"--- Slide OCR Text ---\n{slides}\n\n--- Transcript ---\n{transcript}"
    elif slides:
        combined = f"--- Slide OCR Text ---\n{slides}"
    else:
        combined = transcript  # can be blank
    with open(combined_path, "w", encoding="utf-8") as f:
        f.write(combined)
    return combined

# ---------- Display functions ----------

def render_slides(slide_text):
    slide_blocks = re.split(r'^--- Slide \d+ ---$', slide_text, flags=re.MULTILINE)
    slide_titles = re.findall(r'^--- Slide (\d+) ---$', slide_text, flags=re.MULTILINE)
    slides = [(int(slide_titles[i]), slide_blocks[i+1].strip()) for i in range(len(slide_titles))]
    slides = sorted(slides, key=lambda x: x[0])  # Ensure Slide 1, Slide 2, ...
    for slide_num, content in slides:
        with st.expander(f"Slide {slide_num}", expanded=False):
            bullets = re.findall(r'^[-•]\s*(.+)$', content, flags=re.MULTILINE)
            if bullets:
                st.markdown('\n'.join([f"- {b}" for b in bullets]))
                rest = re.sub(r'^[-•]\s*.+$', '', content, flags=re.MULTILINE).strip()
                if rest:
                    st.markdown(rest.replace('\n', '  \n'))
            else:
                st.markdown(content.replace('\n', '  \n'))

def parse_transcript_segments(transcript_text):
    pattern = re.compile(r'^\[(.*?)\]\s+(.*)$', re.MULTILINE)
    return pattern.findall(transcript_text)

def render_transcript_segments(segments):
    st.markdown(
        """
        <style>
        .transcript-container {
            background: #222;
            padding: 1em;
            border-radius: 0.75em;
            max-height: 380px;
            overflow-y: auto;
            font-size: 1rem;
        }
        .ts-row { display: flex; align-items: flex-start; margin-bottom: .5em; }
        .ts-time { min-width: 72px; color: #63a4fa; font-family: monospace; font-weight: bold; margin-right: 0.75em; }
        .ts-utt { flex: 1 1 auto; color: #ddd; }
        </style>
        """, unsafe_allow_html=True
    )
    st.markdown('<div class="transcript-container">', unsafe_allow_html=True)
    for timestamp, utterance in segments:
        st.markdown(
            f'<div class="ts-row">'
            f'<span class="ts-time">{timestamp}</span>'
            f'<span class="ts-utt">{utterance}</span>'
            f'</div>', unsafe_allow_html=True
        )
    st.markdown('</div>', unsafe_allow_html=True)

def render_transcript(transcript_text):
    segments = parse_transcript_segments(transcript_text)
    if segments:
        render_transcript_segments(segments)
    elif transcript_text.strip():
        st.text_area("Transcript", transcript_text, height=300)
    else:
        st.info("No transcript available.")

def render_combined(combined_text):
    m = re.search(r"--- Slide OCR Text ---\s*(.*?)--- Transcript ---\s*(.*)", combined_text, flags=re.DOTALL)
    if m:
        slide_part, transcript_part = m.group(1).strip(), m.group(2).strip()
    else:
        slides_m = re.search(r"--- Slide OCR Text ---\s*(.*)", combined_text, flags=re.DOTALL)
        transcript_m = re.search(r"--- Transcript ---\s*(.*)", combined_text, flags=re.DOTALL)
        slide_part = slides_m.group(1).strip() if slides_m else ""
        transcript_part = transcript_m.group(1).strip() if transcript_m else (combined_text.strip() if not slides_m else "")
    st.subheader("Slides (OCR)")
    if slide_part:
        render_slides(slide_part)
    else:
        st.info("No slide OCR content available.")
    st.subheader("Transcript")
    if transcript_part:
        render_transcript(transcript_part)
    else:
        st.info("No transcript content available.")

def parse_action_items(raw_text):
    items = []
    raw_text = re.sub(r"^\s*\*\*Actionable.+?Items:\*\*", "", raw_text, flags=re.DOTALL)
    pattern = r'\n\d+\.\s+'
    splits = re.split(pattern, raw_text)
    for block in splits:
        if not block.strip():
            continue
        item = {}
        m = re.search(r'\*\*(.*?)\*\*', block)
        item['Task'] = m.group(1).strip() if m else block.strip().split('\n')[0]
        m = re.search(r'\*\*Owner:\*\*\s*([\s\S]*?)(?=(\*\*Deadline:\*\*|$))', block)
        if m:
            item['Owner'] = m.group(1).strip().replace('**', '')
        m = re.search(r'\*\*Deadline:\*\*\s*([\s\S]*?)(?=(\*\*|$))', block)
        if m:
            item['Deadline'] = m.group(1).strip().replace('**', '')
        m = re.findall(r'- ([^\n]+)', block)
        bullets = [b for b in m if not b.lower().startswith(('owner:', 'deadline:'))]
        if bullets:
            item['Bullets'] = bullets
        items.append(item)
    return items

def render_action_items(items):
    for idx, item in enumerate(items, 1):
        with st.expander(f"{idx}. {item.get('Task', 'Action Item')}"):
            if item.get('Bullets'):
                st.markdown("**Details:**")
                for b in item['Bullets']:
                    st.markdown(f"- {b}")
            if item.get('Owner'):
                st.markdown(f"**Owner:** {item['Owner']}")
            if item.get('Deadline'):
                st.markdown(f"**Deadline:** {item['Deadline']}")

# ---------- Streamlit UI ----------

st.set_page_config(page_title="Multimodal Meeting/Video Summariser", layout="wide")

st.title("📑 Multimodal Meeting & Video Summariser (CM3070 Demo)")

with st.expander("About / How this works", expanded=False):
    st.markdown("""
This tool extracts and summarises meeting content from **audio or video** files.
- **Audio**: Automatic speech recognition (ASR) creates a transcript.
- **Video**: Both the spoken transcript and the slide text (via OCR) are extracted, then combined.
- Generates: **Transcript, Slides (OCR), Combined Input, Summary, Action Items**.
*All processing is local, for privacy-first demo purposes. See 'Limitations' for known edge cases and future work.*
""")

with st.expander("Limitations & Future Work", expanded=False):
    st.warning("""
- **Charts/graphs/images** without text are not extracted by OCR.
- Only English-language content is robustly supported.
- Video processing may take several minutes for long files.
- Future work: Incorporate automated chart/graph summarisation, improve diarisation, expand multi-language support.
""")

os.makedirs("data/samples", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("data/frames", exist_ok=True)

uploaded_file = st.file_uploader(
    "Upload meeting audio or video file (.wav, .mp3, .mp4, .mov, .mkv, etc.)",
    type=["wav", "mp3", "mp4", "mov", "mkv", "avi", "flv", "webm"]
)

if uploaded_file:
    start_time = time.time()   # <<< TIMER START
    input_ext = os.path.splitext(uploaded_file.name)[1].lower()
    input_path = f"data/samples/input{input_ext}"
    save_uploaded_file(uploaded_file, input_path)
    st.success(f"File uploaded: {uploaded_file.name}")

    is_video = is_video_file(input_path)

    transcript_path = "outputs/transcript.txt"
    slides_path = "outputs/slide_texts.txt"
    combined_path = "outputs/combined_transcript.txt"
    summary_path = "outputs/summary.txt"
    action_items_path = "outputs/action_items.txt"

    # Remove previous outputs
    for path in [transcript_path, slides_path, combined_path, summary_path, action_items_path]:
        if os.path.exists(path):
            os.remove(path)
    if os.path.exists("data/frames"):
        shutil.rmtree("data/frames")
        os.makedirs("data/frames", exist_ok=True)

    # --- Progress bar setup ---
    progress = st.progress(0, "Starting pipeline...")

    # 1. Transcription (audio or video)
    with st.spinner("Transcribing audio (Whisper ASR)..."):
        run_subprocess([sys.executable, "core/transcribe.py", input_path], desc="Transcribing audio/video to transcript")
        if not os.path.exists(transcript_path) or os.path.getsize(transcript_path) == 0:
            st.warning("Transcription failed or transcript is empty. Continuing pipeline anyway.")
    progress.progress(1/5, "Step 1/5: Transcription complete.")

    # 2. Slide OCR (if video)
    slides_exist = False
    if is_video:
        with st.spinner("Extracting slides (OCR)..."):
            run_subprocess([sys.executable, "core/video_ocr.py", input_path, "--output", slides_path, "--frames_dir", "data/frames"],
                           desc="Extracting slide text from video frames")
            slides_exist = os.path.exists(slides_path) and os.path.getsize(slides_path) > 0
            if not slides_exist:
                st.warning("Slide OCR failed or no slide text detected. Continuing pipeline anyway.")
    progress.progress(2/5, "Step 2/5: Slide OCR complete.")

    # 3. Combine slides + transcript (handles empty gracefully)
    with st.spinner("Combining transcript and slide text..."):
        slides_path_to_use = slides_path if (is_video and slides_exist) else ""
        combine_transcript_and_slides(transcript_path, slides_path_to_use, combined_path)
    progress.progress(3/5, "Step 3/5: Transcript and slide text combined.")

    # 4. Summarisation
    with st.spinner("Summarising meeting (LLM)..."):
        run_subprocess([sys.executable, "core/summarise.py", combined_path], desc="Generating summary from combined transcript")
        if not os.path.exists(summary_path) or os.path.getsize(summary_path) == 0:
            st.warning("Summary not generated. Continuing pipeline anyway.")
    progress.progress(4/5, "Step 4/5: Meeting summarised.")

    # 5. Action item extraction
    with st.spinner("Extracting action items..."):
        run_subprocess([sys.executable, "core/extract_actions.py", combined_path], desc="Extracting action items from combined transcript")
        if not os.path.exists(action_items_path) or os.path.getsize(action_items_path) == 0:
            st.warning("Action items not generated. Continuing pipeline anyway.")
    progress.progress(5/5, "Step 5/5: Action items extracted.")
    progress.empty()  # remove bar after all done

    # --- Show elapsed time before displaying results ---
    end_time = time.time()
    elapsed = end_time - start_time
    st.success(f"⏱️ Total processing time: {elapsed:.1f} seconds ({int(elapsed//60)} min {int(elapsed%60)} sec)")

    # --- Display results ---
    st.header("Results")
    tabs = st.tabs([
        "Slides (OCR)", "Transcript", "Combined Input", "Summary", "Action Items"
    ])

    # Slides Tab
    with tabs[0]:
        st.markdown("#### Slides (OCR)")
        slides = safe_read(slides_path)
        if slides.strip():
            render_slides(slides)
            st.download_button("Download slide_texts.txt", slides, file_name="slide_texts.txt")
        else:
            st.info("No slides detected (audio input, or OCR did not extract slide text).")

    # Transcript Tab
    with tabs[1]:
        st.markdown("#### Transcript")
        transcript = safe_read(transcript_path)
        render_transcript(transcript)
        st.download_button("Download transcript.txt", transcript, file_name="transcript.txt")

    # Combined Input Tab
    with tabs[2]:
        st.markdown("#### Combined Transcript (Slides + Speech)")
        combined = safe_read(combined_path)
        if combined.strip():
            render_combined(combined)
            st.download_button("Download combined_transcript.txt", combined, file_name="combined_transcript.txt")
        else:
            st.info("No combined input available.")

    # Summary Tab
    with tabs[3]:
        st.markdown("#### Meeting Summary")
        summary = safe_read(summary_path)
        if summary.strip():
            with st.expander("Show Summary", expanded=True):
                st.markdown(summary.replace('\n', '  \n'))
            st.download_button("Download summary.txt", summary, file_name="summary.txt")
        else:
            st.info("No summary generated.")

    # Action Items Tab
    with tabs[4]:
        st.markdown("#### Action Items")
        raw_action_text = safe_read(action_items_path)
        action_items = parse_action_items(raw_action_text)
        if action_items:
            render_action_items(action_items)
        else:
            st.info("No action items detected or unable to parse.")
        st.download_button("Download action_items.txt", raw_action_text, file_name="action_items.txt")

else:
    st.info("Please upload a meeting audio or video file to start.")
