# core/extract_actions.py

"""
Extracts action items from a meeting transcript using IBM Granite 3.2 8B Instruct.
- Removes timestamps for better LLM extraction.
- Uses overlapping chunks for context coherence.
- Aggregates, deduplicates, and cleans action items for professional output.
"""

import sys
import re
from transformers import pipeline

CHUNK_SIZE = 4000      # Characters per chunk
CHUNK_OVERLAP = 500    # Overlap for context (in characters)
MAX_ACTION_TOKENS = 200

def load_text(path):
    """Load transcript text from a file."""
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

def save_actions(actions, path):
    """Save extracted action items to a file, stripped of whitespace."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(actions.strip())

def chunk_with_overlap(text, chunk_size, overlap):
    """Split text into overlapping chunks for more coherent extraction."""
    if chunk_size <= overlap:
        raise ValueError("Chunk size must be greater than overlap.")
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def extract_action_items_from_chat(result):
    """
    Extract the assistant's reply (action items) from Granite's chat output.
    Returns the assistant's content string.
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

def clean_action_items(raw_actions):
    """
    Cleans up LLM-generated action items:
    - Removes empty bullets, 'Task:'-only lines, and duplicates.
    """
    lines = raw_actions.splitlines()
    cleaned = []
    seen = set()
    for line in lines:
        txt = line.strip('-• ').strip()
        # Ignore empty, generic, or placeholder lines
        if not txt or txt.lower() in {
            "task", "no action items found.", "no action items or tasks identified"
        }:
            continue
        # Only keep unique tasks
        if txt not in seen:
            cleaned.append(line.strip())
            seen.add(txt)
    return '\n'.join(cleaned)

def extract_action_items(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP, max_tokens=MAX_ACTION_TOKENS):
    """
    Extracts action items as bullet list, aggregating and deduplicating results.
    """
    summarizer = pipeline(
        "text-generation",
        model="ibm-granite/granite-3.2-8b-instruct",
        device=0
    )
    chunks = chunk_with_overlap(text, chunk_size, overlap)
    all_actions = []

    instruction = (
        "Extract all clear, concrete action items or tasks from the following meeting transcript. "
        "For each, specify the task, owner (if possible), and any deadline. Output as a bullet list. "
        "If there are no action items, say: 'No action items found.' Do not invent tasks or repeat previous items.\n\n"
    )
    for idx, chunk in enumerate(chunks):
        prompt = instruction + chunk
        messages = [{"role": "user", "content": prompt}]
        print(f"Processing chunk {idx+1}/{len(chunks)}, length {len(chunk)}")
        try:
            result = summarizer(messages, max_new_tokens=max_tokens)
            actions = extract_action_items_from_chat(result)
            if "-" in actions or "•" in actions:
                all_actions.append(actions)
        except Exception as e:
            all_actions.append(f"Error on chunk {idx+1}: {e}")
            continue

    # Combine, clean, and deduplicate action items
    raw_actions = "\n".join(all_actions)
    cleaned_actions = clean_action_items(raw_actions)

    # (Optional) Final aggregation step: if many items, ask LLM to deduplicate and rewrite.
    if len(cleaned_actions.splitlines()) > 12:
        agg_prompt = (
            "Deduplicate, clarify, and rewrite the following list of meeting action items. "
            "Group similar tasks, remove repeated or vague entries, and ensure each bullet is actionable, concrete, and assigned where possible:\n\n"
            + cleaned_actions
        )
        agg_messages = [{"role": "user", "content": agg_prompt}]
        agg_result = summarizer(agg_messages, max_new_tokens=MAX_ACTION_TOKENS * 2)
        final_actions = extract_action_items_from_chat(agg_result)
        return clean_action_items(final_actions)
    else:
        return cleaned_actions

if __name__ == "__main__":
    # Usage: python extract_actions.py <transcript.txt>
    if len(sys.argv) < 2:
        print("Usage: python extract_actions.py <transcript.txt>")
        sys.exit(1)
    transcript_path = sys.argv[1]
    actions_path = "outputs/action_items.txt"
    text = load_text(transcript_path)
    text = strip_timestamps(text)
    actions = extract_action_items(text)
    save_actions(actions, actions_path)
    print(f"Action items saved to {actions_path}")
