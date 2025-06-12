# core/summarise.py
"""
Usage: python core/summarise.py
Reads outputs/transcript.json and writes
outputs/summary.txt & outputs/actions.txt
"""
import json, re, torch
from transformers import pipeline

# Load transcript
transcript = json.load(open("outputs/transcript.json"))["text"][:4096]

# Choose device: GPU 0 if available, otherwise CPU
device = 0 if torch.cuda.is_available() else -1

summariser = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    device=device
)

prompt = (
    "Summarise the following meeting transcript in 5 sentences. "
    "Afterwards list any explicit decisions or action items with the person responsible, "
    "each on a new line starting with a dash:\n\n"
    + transcript
)

out = summariser(prompt, max_length=220, min_length=150, do_sample=False)[0]["summary_text"]

# Split into summary + actions
parts = re.split(r"(?i)action items?:", out, maxsplit=1)
summary = parts[0].strip()
actions = parts[1].strip() if len(parts) == 2 else "No clear action items detected."

# Write outputs
with open("outputs/summary.txt", "w") as f:
    f.write(summary)

with open("outputs/actions.txt", "w") as f:
    f.write(actions)

print("Summary + actions written to outputs/")
