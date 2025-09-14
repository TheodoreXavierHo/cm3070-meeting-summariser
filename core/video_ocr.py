# core/video_ocr.py

"""
Extract unique slide texts from a meeting video, optimised for speed and high-quality, information-dense output.
- Step 1: Extract frames at regular intervals.
- Step 2: Pre-deduplicate frames using image hashing (skip near-duplicates).
- Step 3: OCR only unique frames (EasyOCR, GPU if available).
- Step 4: Deduplicate slide texts (fuzzy/content-based overlap).
- Step 5: Cluster by topic and keep only the most information-rich slide per topic.
- Output is ready for multimodal summarisation.
"""

import cv2
import easyocr
import os
import sys
import argparse
from typing import List
from PIL import Image
import imagehash
from rapidfuzz import fuzz

def extract_video_frames(video_path: str, output_dir: str = "frames", interval: int = 5) -> List[str]:
    """
    Step 1: Extract frames every `interval` seconds from video.
    Returns list of frame file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = int(frame_count / fps) if fps else 0
    frame_paths = []
    if not fps or not frame_count:
        print(f"Error: Unable to read video {video_path}.")
        return frame_paths

    print(f"Video duration: {duration}s at {fps:.2f} FPS; extracting a frame every {interval}s...")
    for sec in range(0, duration, interval):
        vidcap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        success, image = vidcap.read()
        if success:
            fname = os.path.join(output_dir, f"frame_{sec:04d}.jpg")
            cv2.imwrite(fname, image)
            frame_paths.append(fname)
            print(f"Extracted frame at {sec}s -> {fname}")
        else:
            print(f"Warning: Could not extract frame at {sec}s")
    vidcap.release()
    print(f"Extracted {len(frame_paths)} frames.")
    return frame_paths

def remove_duplicate_frames_by_hash(frame_files: List[str], hash_threshold: int = 5) -> List[str]:
    """
    Step 2: Remove near-duplicate images using image hashing.
    Returns list of visually unique frame paths.
    """
    unique_files = []
    last_hash = None
    for frame_path in frame_files:
        img = Image.open(frame_path)
        img_hash = imagehash.average_hash(img, hash_size=8)
        if last_hash is None or abs(img_hash - last_hash) >= hash_threshold:
            unique_files.append(frame_path)
            last_hash = img_hash
        else:
            print(f"Skipped visually duplicate frame: {frame_path}")
    print(f"Pre-deduplicated: {len(frame_files)} -> {len(unique_files)} frames (before OCR).")
    return unique_files

def perform_ocr_on_frames(frame_files: List[str], lang: str = 'en', gpu: bool = True) -> List[str]:
    """
    Step 3: Run EasyOCR on unique frames; returns list of OCR results (strings).
    """
    # Suppress verbose progress bar (causes Unicode issues on Windows consoles)
    reader = easyocr.Reader([lang], gpu=gpu, verbose=False)

    slide_texts = []
    for idx, frame_path in enumerate(frame_files):
        try:
            results = reader.readtext(frame_path, detail=0, paragraph=True)
            text = "\n".join(results).strip()
            slide_texts.append(text)
            print(f"OCR [{idx+1}/{len(frame_files)}] {frame_path}: {len(text)} chars")
        except Exception as e:
            print(f"Error on OCR for {frame_path}: {e}")
            slide_texts.append("")
    return slide_texts

def deduplicate_slides_by_similarity(slide_texts, min_change=20, fuzzy_threshold=90):
    """
    Step 4a: Deduplicate progressive slides using fuzzy string similarity and simple size check.
    - Keeps only slides with substantial change vs previous.
    """
    deduped = []
    last = ""
    for idx, text in enumerate(slide_texts):
        if not text.strip():
            continue
        is_unique = True
        if last:
            if fuzz.ratio(text, last) >= fuzzy_threshold:
                is_unique = False
            elif text.startswith(last) and len(text) - len(last) < min_change:
                is_unique = False
        if is_unique:
            deduped.append(text)
        last = text
    print(f"Fuzzy deduplicated: {len(slide_texts)} -> {len(deduped)} slides.")
    return deduped

def deduplicate_slides_by_content(slide_texts, word_overlap_threshold=0.85, line_overlap_threshold=0.85):
    """
    Step 4b: Further deduplicate slides based on bag-of-words and line overlap.
    - Filters incremental builds by ignoring slides with too much overlap.
    """
    deduped = []
    prev_words = set()
    prev_lines = set()
    for idx, text in enumerate(slide_texts):
        words = set(text.lower().split())
        lines = set([l.strip().lower() for l in text.splitlines() if l.strip()])
        word_overlap = len(words & prev_words) / (len(words | prev_words) or 1)
        line_overlap = len(lines & prev_lines) / (len(lines | prev_lines) or 1)
        if (word_overlap < word_overlap_threshold) and (line_overlap < line_overlap_threshold):
            deduped.append(text)
            prev_words = words
            prev_lines = lines
        else:
            print(f"Skipped incremental slide at {idx+1} (overlap: word={word_overlap:.2f}, line={line_overlap:.2f})")
    print(f"Content deduplicated: {len(slide_texts)} -> {len(deduped)} slides.")
    return deduped

def select_most_informative_slide_per_topic(slide_texts, topic_prefix_len=3, min_length=10, global_fuzzy=85):
    """
    Step 5: 
    - Cluster by topic (first N words of slide).
    - For each topic, keep the slide with the most words (most complete).
    - Skip any slide globally >global_fuzzy% similar to an already kept slide.
    """
    from collections import defaultdict
    kept = []
    topics = defaultdict(list)
    # Step 1: Cluster by topic prefix (first n words)
    for text in slide_texts:
        words = text.lower().split()
        if not words:
            continue
        topic = " ".join(words[:topic_prefix_len])
        topics[topic].append(text)
    # Step 2: For each topic, keep longest slide
    longest_per_topic = []
    for group in topics.values():
        best_slide = max(group, key=lambda s: len(s.split()))
        if len(best_slide.split()) >= min_length:
            longest_per_topic.append(best_slide)
    # Step 3: Global fuzzy filter (no two slides >global_fuzzy% similar)
    for slide in sorted(longest_per_topic, key=lambda s: -len(s.split())):
        if all(fuzz.ratio(slide, s) < global_fuzzy for s in kept):
            kept.append(slide)
    print(f"Final topic clustering: reduced {len(slide_texts)} to {len(kept)} slides.")
    return kept

def save_slide_texts(slide_texts: List[str], out_path: str = "outputs/slide_texts.txt"):
    """
    Write unique slide texts to a file with clear separators.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for idx, slide in enumerate(slide_texts):
            f.write(f"--- Slide {idx+1} ---\n{slide}\n\n")
    print(f"Saved unique slide texts to {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Extract unique, information-dense slide texts from a video using EasyOCR, with topic-based deduplication.")
    parser.add_argument("video_file", help="Path to video file (mp4, mov, etc.)")
    parser.add_argument("--interval", type=int, default=5, help="Seconds between frames (default: 5)")
    parser.add_argument("--lang", type=str, default='en', help="OCR language (default: 'en')")
    parser.add_argument("--output", type=str, default="outputs/slide_texts.txt", help="Output text file")
    parser.add_argument("--frames_dir", type=str, default="data/frames", help="Directory for extracted frames")
    parser.add_argument("--hash_threshold", type=int, default=5, help="Image hash difference threshold (lower=more strict)")
    parser.add_argument("--min_change", type=int, default=20, help="Minimum character change to keep a slide")
    parser.add_argument("--fuzzy_threshold", type=int, default=90, help="Fuzzy similarity threshold for deduplication")
    parser.add_argument("--word_overlap", type=float, default=0.85, help="Word overlap threshold (0.0-1.0)")
    parser.add_argument("--line_overlap", type=float, default=0.85, help="Line overlap threshold (0.0-1.0)")
    parser.add_argument("--topic_prefix_len", type=int, default=3, help="Topic prefix word count for clustering")
    parser.add_argument("--topic_min_length", type=int, default=10, help="Min words to keep in topic clustering")
    parser.add_argument("--topic_global_fuzzy", type=int, default=85, help="Fuzzy threshold for global duplicate skip")
    parser.add_argument("--cpu", action="store_true", help="Use CPU even if GPU is available")
    args = parser.parse_args()

    # Step 1: Extract frames
    frame_files = extract_video_frames(args.video_file, args.frames_dir, interval=args.interval)
    # Step 2: Image hash deduplication
    frame_files = remove_duplicate_frames_by_hash(frame_files, hash_threshold=args.hash_threshold)
    # Step 3: OCR unique frames
    slide_texts = perform_ocr_on_frames(frame_files, lang=args.lang, gpu=not args.cpu)
    # Step 4a: Fuzzy deduplication
    slide_texts = deduplicate_slides_by_similarity(slide_texts, min_change=args.min_change, fuzzy_threshold=args.fuzzy_threshold)
    # Step 4b: Content deduplication (word/line overlap)
    slide_texts = deduplicate_slides_by_content(
        slide_texts, word_overlap_threshold=args.word_overlap, line_overlap_threshold=args.line_overlap
    )
    # Step 5: Topic clustering and information-maximal selection (final filter!)
    slide_texts = select_most_informative_slide_per_topic(
        slide_texts,
        topic_prefix_len=args.topic_prefix_len,
        min_length=args.topic_min_length,
        global_fuzzy=args.topic_global_fuzzy,
    )
    # Save to output file
    save_slide_texts(slide_texts, out_path=args.output)

    print("OCR pipeline complete.")
    print(f"Total frames processed: {len(frame_files)}; Unique slides: {len(slide_texts)}.")

if __name__ == "__main__":
    main()
