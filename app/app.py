import streamlit as st
import os
import subprocess
import sys  # Ensures subprocesses use the venv Python

st.title("Multimodal Meeting Summariser (CM3070 Prototype)")
st.write("Upload a meeting audio file to generate a transcript, summary, and action items.")

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

    # Transcript
    with tabs[0]:
        try:
            with open("outputs/transcript.txt", "r", encoding="utf-8") as f:
                st.text_area("Transcript", f.read(), height=300)
        except FileNotFoundError:
            st.error("Transcript not found.")

    # Summary
    with tabs[1]:
        try:
            with open("outputs/summary.txt", "r", encoding="utf-8") as f:
                st.text_area("Summary", f.read(), height=200)
        except FileNotFoundError:
            st.error("Summary not found.")

    # Action Items
    with tabs[2]:
        try:
            with open("outputs/action_items.txt", "r", encoding="utf-8") as f:
                st.text_area("Action Items", f.read(), height=200)
        except FileNotFoundError:
            st.error("Action items not found.")

    # Download buttons
    st.markdown("### Download outputs")
    for outname in ["transcript.txt", "summary.txt", "action_items.txt"]:
        output_file = f"outputs/{outname}"
        if os.path.exists(output_file):
            with open(output_file, "rb") as f:
                st.download_button(
                    f"Download {outname}", f, file_name=outname
                )
else:
    st.info("Please upload a meeting audio file to start.")
