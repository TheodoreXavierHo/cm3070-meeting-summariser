import streamlit as st
import os
import subprocess
import sys  # Ensures subprocesses use the venv Python

st.title("Multimodal Meeting Summariser (CM3070 Prototype)")
st.markdown(
    "> **Prototype Notice:** This tool is a functional demo. "
    "Summaries and action items may be incomplete or imperfect for long or complex meetings. "
    "See report for limitations and future improvements."
)

# Ensure data and outputs directories exist
os.makedirs("data/samples", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

uploaded_file = st.file_uploader("Upload meeting audio (.wav or .mp3)", type=["wav", "mp3"])
if uploaded_file:
    input_path = f"data/samples/{uploaded_file.name}"

    # Remove file if it already exists to avoid permission error
    if os.path.exists(input_path):
        os.remove(input_path)

    # Save uploaded file
    with open(input_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success(f"Uploaded file: {uploaded_file.name}")

    # --- Run ASR (transcribe.py) ---
    with st.spinner("Transcribing audio (Whisper)..."):
        result = subprocess.run(
            [sys.executable, "core/transcribe.py", input_path],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            st.error(f"Transcription failed: {result.stderr}")
            st.stop()

    # --- Run summariser (summarise.py) ---
    with st.spinner("Summarising transcript (BART)..."):
        result = subprocess.run(
            [sys.executable, "core/summarise.py", "outputs/transcript.txt"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            st.error(f"Summarisation failed: {result.stderr}")
            st.stop()

    # --- Run action item extractor (extract_actions.py) ---
    with st.spinner("Extracting action items..."):
        result = subprocess.run(
            [sys.executable, "core/extract_actions.py", "outputs/transcript.txt"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            st.error(f"Action extraction failed: {result.stderr}")
            st.stop()

    # --- Display results ---
    st.header("Results")
    tabs = st.tabs(["Transcript", "Summary", "Action Items"])

    # Transcript Tab
    with tabs[0]:
        try:
            with open("outputs/transcript.txt", "r", encoding="utf-8") as f:
                transcript = f.read()
            with st.expander("Show Transcript", expanded=True):
                st.text_area("Transcript", transcript, height=300)
            st.download_button("Download transcript.txt", transcript, file_name="transcript.txt")
        except FileNotFoundError:
            st.error("Transcript not found.")

    # Summary Tab
    with tabs[1]:
        try:
            with open("outputs/summary.txt", "r", encoding="utf-8") as f:
                summary = f.read()
            with st.expander("Show Summary", expanded=True):
                st.markdown("#### 150-word summary\n")
                st.markdown(summary.replace('\n', '  \n'))
            st.download_button("Download summary.txt", summary, file_name="summary.txt")
        except FileNotFoundError:
            st.error("Summary not found.")

    # Action Items Tab
    with tabs[2]:
        try:
            with open("outputs/action_items.txt", "r", encoding="utf-8") as f:
                actions = f.read()
            with st.expander("Show Action Items", expanded=True):
                st.markdown("#### Action Items (task – owner – due date)\n")
                # Try to pretty print bullets if present
                if "-" in actions or "*" in actions:
                    st.markdown(actions.replace('-', '\n-').replace('*', '\n*'))
                else:
                    st.text_area("Action Items", actions, height=200)
            st.download_button("Download action_items.txt", actions, file_name="action_items.txt")
        except FileNotFoundError:
            st.error("Action items not found.")

else:
    st.info("Please upload a meeting audio file to start.")
