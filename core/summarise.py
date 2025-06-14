# core/summarise.py
import sys
from transformers import pipeline

def load_transcript(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def save_summary(summary, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write(summary.strip())

def summarise_text(text, max_tokens=150):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0)
    # Split text if it's very long (BART limit is ~1024 tokens)
    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
    summaries = []
    for chunk in chunks:
        result = summarizer(chunk, max_length=max_tokens, min_length=60, do_sample=False)
        summaries.append(result[0]["summary_text"])
    return "\n".join(summaries)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python summarise.py <transcript.txt>")
        sys.exit(1)
    transcript_path = sys.argv[1]
    summary_path = "outputs/summary.txt"
    text = load_transcript(transcript_path)
    summary = summarise_text(text)
    save_summary(summary, summary_path)
    print("Summary saved to outputs/summary.txt")
