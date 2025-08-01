# core/summarise.py

"""
Summarises a meeting transcript using IBM Granite 3.2 8B Instruct (chat interface).
- Removes timestamps for better LLM summarisation.
- Uses overlapping chunks for context coherence.
- Aggregates chunk summaries and generates a final concise output within the specified word limit.
- Extracts only the model's assistant reply as the final summary.
"""

import sys
import re
from transformers import pipeline

WORD_LIMIT = 500       # Target summary length in words
CHUNK_SIZE = 4000      # Characters per chunk (adjust for VRAM/context window)
CHUNK_OVERLAP = 500    # Overlap between chunks (in characters)

def load_transcript(path):
    """Load a text file as a string."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def strip_timestamps(text):
    """Remove '[start - end]' timestamps from each line."""
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        cleaned_line = re.sub(r"^\[\d+\.\d+s\s*-\s*\d+\.\d+s\]\s*", "", line)
        if cleaned_line:
            cleaned.append(cleaned_line.strip())
    return "\n".join(cleaned)

def save_summary(summary, path):
    """Save summary string to a file (stripped of excess whitespace)."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(summary.strip())

def extract_summary_from_chat(result):
    """
    Extract the assistant's response (summary) from Granite's chat output structure.
    """
    if (
        isinstance(result, list)
        and result
        and isinstance(result[0], dict)
        and "generated_text" in result[0]
    ):
        messages = result[0]["generated_text"]
        if isinstance(messages, list):
            for msg in messages:
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    return msg.get("content", "").strip()
    return str(result).strip()

def chunk_with_overlap(text, chunk_size, overlap):
    """Split text into overlapping chunks for more coherent summarisation."""
    if chunk_size <= overlap:
        raise ValueError("Chunk size must be greater than overlap.")
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def count_words(text):
    return len(re.findall(r'\w+', text))

def truncate_to_word_limit(text, word_limit):
    """Truncate the summary to the specified word limit, not splitting sentences."""
    words = text.split()
    if len(words) <= word_limit:
        return text
    # Try to cut at a sentence boundary
    truncated = ' '.join(words[:word_limit])
    # Try to end at the last period if possible
    last_period = truncated.rfind('.')
    if last_period > 0:
        return truncated[:last_period+1]
    return truncated

def summarise_text(text, word_limit=WORD_LIMIT, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Summarises the transcript using overlapping chunks and final aggregation.
    """
    summarizer = pipeline(
        "text-generation",
        model="ibm-granite/granite-3.2-8b-instruct",
        device=0
    )

    # 1. Chunk the transcript with overlap
    chunks = chunk_with_overlap(text, chunk_size, overlap)
    chunk_summaries = []

    for idx, chunk in enumerate(chunks):
        prompt = (
            f"Summarise the following meeting transcript in no more than {word_limit} words. "
            "Be concise, factual, and avoid repetition. Highlight only the most important discussion points, decisions, and action items.\n"
            + chunk
        )
        messages = [{"role": "user", "content": prompt}]
        try:
            result = summarizer(messages, max_new_tokens=word_limit * 2)
            summary = extract_summary_from_chat(result)
            if summary:
                chunk_summaries.append(summary)
            else:
                chunk_summaries.append(f"No summary generated for chunk {idx+1}.")
        except Exception as e:
            chunk_summaries.append(f"Error on chunk {idx+1}: {e}")

    # 2. Aggregate and re-summarise if there are multiple chunks
    if len(chunk_summaries) == 1:
        final_summary = chunk_summaries[0]
    else:
        combined = "\n".join(chunk_summaries)
        agg_prompt = (
            f"Combine and rewrite the following draft summaries into a single, cohesive meeting summary of no more than {word_limit} words. "
            "Remove repeated points, ensure logical flow, and avoid splitting sentences."
            "\n\n" + combined
        )
        agg_messages = [{"role": "user", "content": agg_prompt}]
        agg_result = summarizer(agg_messages, max_new_tokens=word_limit * 2)
        final_summary = extract_summary_from_chat(agg_result)

    # 3. Truncate for absolute word limit, do not cut sentences
    final_summary = truncate_to_word_limit(final_summary, word_limit)
    return final_summary

if __name__ == "__main__":
    # Usage: python summarise.py <transcript.txt>
    if len(sys.argv) < 2:
        print("Usage: python summarise.py <transcript.txt>")
        sys.exit(1)
    transcript_path = sys.argv[1]
    summary_path = "outputs/summary.txt"
    text = load_transcript(transcript_path)
    text = strip_timestamps(text)
    summary = summarise_text(text)
    save_summary(summary, summary_path)
    print(f"Summary saved to {summary_path}")
