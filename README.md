# Multimodal Meeting & Video Summariser

## *CM3070 Final Year Project (Template 4.1)*

> Local, privacy-first pipeline for summarising meetings, lectures, and presentations from audio and video files.
> Built for University of London (UOL) BSc Computer Science (Machine Learning & AI) – SIM Global Education.

---

## Features

* **Supports both audio and video files** (.wav, .mp3, .mp4, .mov, .mkv, .avi, .flv, .webm)
* **Automatic Speech Recognition (ASR):**
  Whisper/Faster-Whisper transcribes speech to text, including timestamps.
* **Slide OCR for Video:**
  Extracts slide text from presentation videos using EasyOCR and image deduplication.
* **Multimodal Summarisation:**
  Combines transcript and slide text, summarised by a local LLM (Granite-3.2-8B Instruct).
* **Action Item Extraction:**
  Extracts actionable tasks, owners, and deadlines using LLM-based prompts.
* **User-Friendly Web UI:**
  Streamlit app for upload, processing, and download of all results.
* **Full local processing:**
  All computation runs locally for privacy; no data leaves your machine.
* **Professional formatting:**
  Expandable slides, YouTube-style transcripts, downloadable results, progress bar, total run timer.

---

## System Overview

```mermaid
flowchart TD
    A[Audio/Video Upload]
    B[Transcribe Speech (Whisper)]
    C{Is Video?}
    D[Extract Frames & OCR Slides (EasyOCR)]
    E[Slide Text]
    F[Transcript]
    G[Combine Transcript & Slides]
    H[Summarise (Granite-3.2-8B)]
    I[Extract Action Items (Granite-3.2-8B)]
    J[Summary]
    K[Action Items]

    A --> B
    B --> F
    A --> C
    C -- No --> F
    C -- Yes --> D
    D --> E
    E --> G
    F --> G
    G --> H
    G --> I
    H --> J
    I --> K
```

---

## Usage

1. **Install dependencies**
   (Recommended: Python 3.9+, with `pip` and `virtualenv`)

   ```sh
   pip install -r requirements.txt
   ```

   * For OCR: [EasyOCR](https://github.com/JaidedAI/EasyOCR), [OpenCV](https://opencv.org/), [imagehash](https://github.com/JohannesBuchner/imagehash)
   * For ASR: [faster-whisper](https://github.com/SYSTRAN/faster-whisper), [ffmpeg](https://ffmpeg.org/) (must be installed and on your PATH)
   * For LLM: [transformers](https://huggingface.co/docs/transformers/index)
   * For UI: [streamlit](https://streamlit.io/)

2. **Start the app**

   ```sh
   streamlit run app/app.py
   ```

3. **Upload your meeting audio or video file.**

   * All processing will run locally; output files appear in `outputs/`.
   * UI displays transcript, extracted slides, combined input, summary, action items, with download buttons for each.

---

## File Structure

``` md
project-root/
│
├─ app/
│   └─ app.py             # Streamlit frontend
│
├─ core/
│   ├─ transcribe.py      # Audio/video → transcript (Whisper)
│   ├─ video_ocr.py       # Video → slide text (EasyOCR + deduplication)
│   ├─ summarise.py       # Transcript (+slides) → summary (Granite-3.2-8B)
│   ├─ extract_actions.py # Extract action items (Granite-3.2-8B)
│   └─ pipeline.py        # CLI batch runner
│
├─ outputs/               # All result files (auto-created)
├─ data/samples/          # Uploaded user files (auto-created)
├─ data/frames/           # Video frames for OCR (auto-cleared)
├─ requirements.txt
├─ README.md
└─ .streamlit/config.toml # (optional) increase maxUploadSize if needed
```

---

## Model and Pipeline Details

* **Speech Recognition:**
  *Faster-Whisper* (`large-v2`, CUDA or CPU fallback)

  * Outputs timestamped transcript

* **Summarisation / Action Extraction:**
  *IBM Granite-3.2-8B-Instruct* (local via HuggingFace/transformers)

  * Sliding window + aggregation for long context
  * Final summary limited to 500 words (configurable)
  * Action items deduplicated, owner/deadline extracted if possible

* **OCR:**
  *EasyOCR* (GPU if available, falls back to CPU)

  * Image deduplication by perceptual hash, fuzzy text similarity, topic clustering

* **UI:**
  *Streamlit*

  * Results: Expandable slide cards, YouTube-style transcript, progress bar, timer

---

## Privacy & Ethics

* **All computation is local.**
  No data is sent to external servers; user uploads are never logged or shared.
* **Model weights** are downloaded from public sources; check [HuggingFace T\&Cs](https://huggingface.co/ibm-granite/granite-3.2-8b-instruct) for licensing.
* **User responsibility:** Ensure any meeting or video files are used in compliance with organisational policy and data privacy laws.

---

## Limitations & Future Work

* **OCR only extracts visible text**—charts, graphs, and images without readable text are not included.
* **English language only** for best results (other languages are untested).
* **Latency:** Video OCR and LLM summarisation are computationally expensive.
* **Slide detection** uses content deduplication; slide transitions or complex animations may reduce accuracy.
* **Planned improvements:**

  * Automated chart/graph summarisation
  * Diarisation (speaker attribution)
  * Multi-language support
  * PDF and CSV export
  * GPU auto-detection and settings UI

---

## Citation & Credits

* Whisper, Faster-Whisper, EasyOCR, Transformers by HuggingFace, IBM Granite-3.2-8B-Instruct

---

## Quick Start Tips

* **Increase file upload size:**
  Edit `.streamlit/config.toml` as follows to allow >200MB files:

  ``` md
  [server]
  maxUploadSize = 1024
  ```

* **GPU recommended** for large models (check CUDA setup for PyTorch and EasyOCR).
* **Clear outputs:** All result files are written to the `outputs/` folder.
