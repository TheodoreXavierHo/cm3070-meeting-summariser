# core/extract_actions.py
import sys
from transformers import pipeline

def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def save_actions(actions, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write(actions.strip())

def chunk_text(text, chunk_size=1500):
    # Non-overlapping chunks
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def extract_action_items(text):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0)
    chunks = chunk_text(text, chunk_size=1500)
    all_actions = []
    
    # Use full prompt only once, for the first chunk
    instruction = (
        "Extract all action items or tasks from the following meeting transcript. "
        "For each action, specify the task, owner (if possible), and any deadline mentioned. "
        "Output as a bullet point list.\n\n"
    )
    for i, chunk in enumerate(chunks):
        if i == 0:
            prompt = instruction + chunk
        else:
            prompt = "Continue extracting action items from the next part of the transcript:\n" + chunk
        print(f"Processing chunk {i+1}/{len(chunks)}, length {len(chunk)}")
        try:
            result = summarizer(prompt, max_length=150, min_length=40, do_sample=False)
            # Only append if model actually produced a bullet list (basic check)
            if "-" in result[0]["summary_text"] or "•" in result[0]["summary_text"]:
                all_actions.append(result[0]["summary_text"])
        except Exception as e:
            print(f"Error on chunk {i+1}: {e}")
            continue
    return "\n".join(all_actions)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_actions.py <transcript.txt>")
        sys.exit(1)
    transcript_path = sys.argv[1]
    actions_path = "outputs/action_items.txt"
    text = load_text(transcript_path)
    actions = extract_action_items(text)
    save_actions(actions, actions_path)
    print("Action items saved to outputs/action_items.txt")
