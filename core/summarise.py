# core/summarise.py

"""
Summarises a meeting transcript using IBM Granite 3.3 Instruct.
- Overlapping chunks + final aggregation
- Auto-picks 8B vs 2B based on GPU VRAM and quantisation
- Optional 4/8-bit quant with GPU offload; safe CPU fallback on OOM
"""

import os
import re
import sys
from typing import List, Tuple

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# ====== knobs ======
WORD_LIMIT = 500           # target words
CHUNK_SIZE = 4000          # chars per chunk
CHUNK_OVERLAP = 500        # chars overlap

# ====== Granite 3.3 models ======
GRANITE_8B = "ibm-granite/granite-3.3-8b-instruct"
GRANITE_2B = "ibm-granite/granite-3.3-2b-instruct"

# ====== env overrides ======
# FYP_MODEL_ID      -> force a specific HF model id (skips auto-pick)
# FYP_FORCE_MODEL   -> "8b" | "2b" to force family
# FYP_QUANT         -> "auto" | "fp16" | "8bit" | "4bit"   (default: "4bit")
ENV_MODEL_ID = os.getenv("FYP_MODEL_ID", "").strip()
FORCE_FAMILY = os.getenv("FYP_FORCE_MODEL", "").strip().lower()   # "8b"|"2b"|""(auto)
DEFAULT_QUANT = os.getenv("FYP_QUANT", "4bit").strip().lower()

# ====== I/O helpers ======
def load_transcript(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def save_summary(summary: str, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(summary.strip())

def strip_timestamps(text: str) -> str:
    """Remove leading '[12.3s - 45.6s] ' style timestamps from lines."""
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        cleaned_line = re.sub(r"^\[\d+\.\d+s\s*-\s*\d+\.\d+s\]\s*", "", line)
        if cleaned_line:
            cleaned.append(cleaned_line.strip())
    return "\n".join(cleaned)

# ====== text utils ======
def chunk_with_overlap(text: str, chunk_size: int, overlap: int) -> List[str]:
    if chunk_size <= overlap:
        raise ValueError("Chunk size must be greater than overlap.")
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def truncate_to_word_limit(text: str, word_limit: int) -> str:
    words = text.split()
    if len(words) <= word_limit:
        return text
    truncated = " ".join(words[:word_limit])
    last_period = truncated.rfind(".")
    return truncated[: last_period + 1] if last_period > 0 else truncated

# ====== device / VRAM helpers ======
def _cuda_vram_gb() -> Tuple[bool, float]:
    if not torch.cuda.is_available():
        return False, 0.0
    try:
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        return True, float(total)
    except Exception:
        return False, 0.0

def _choose_family(quant: str) -> str:
    """
    Your policy:
      - VRAM ≤ 6 GB  -> 2B
      - VRAM 7–11 GB -> 2B, unless quant == "4bit" and VRAM ≥ 8 GB -> 8B
      - VRAM ≥ 12 GB -> 8B
    Overrides: ENV_MODEL_ID, FORCE_FAMILY
    """
    if ENV_MODEL_ID:
        return ENV_MODEL_ID
    if FORCE_FAMILY in ("8b", "2b"):
        return GRANITE_8B if FORCE_FAMILY == "8b" else GRANITE_2B

    has_cuda, vram_gb = _cuda_vram_gb()
    if not has_cuda:
        return GRANITE_2B

    if vram_gb <= 6.0:
        return GRANITE_2B

    if 7.0 <= vram_gb < 12.0:
        if quant.lower() == "4bit" and vram_gb >= 8.0:
            return GRANITE_8B
        return GRANITE_2B

    # vram >= 12
    return GRANITE_8B

# ====== pipeline builder ======
def _build_textgen_pipeline(quant: str = DEFAULT_QUANT,
                            gpu_mem_gb: int = 14,
                            max_new_tokens: int = 320):
    """
    quant:
      - "4bit"/"8bit": try bitsandbytes quant on CUDA with offload
      - "fp16": fp16 on CUDA (if enough VRAM) else CPU
      - "auto": same as "fp16" but decide device automatically
    """
    model_id = _choose_family(quant)
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    has_cuda, vram_gb = _cuda_vram_gb()
    # allow CUDA from 4 GB upward so 2B-4bit can run on small GPUs
    want_cuda = has_cuda and vram_gb >= 4.0

    device_map = "auto" if want_cuda else None
    # Soft cap VRAM for offload; keep a little headroom
    cap = max(4, int(min(vram_gb - 2, gpu_mem_gb))) if want_cuda else 0
    max_memory = {0: f"{cap}GiB", "cpu": "64GiB"} if want_cuda else None

    offload_dir = os.path.join(os.getcwd(), "offload_cache")
    os.makedirs(offload_dir, exist_ok=True)

    # Quantised paths first (least VRAM)
    if quant in ("4bit", "8bit") and want_cuda:
        try:
            from transformers import BitsAndBytesConfig
            bnb = BitsAndBytesConfig(
                load_in_4bit=(quant == "4bit"),
                load_in_8bit=(quant == "8bit"),
                bnb_4bit_use_double_quant=True if quant == "4bit" else None,
                bnb_4bit_quant_type="nf4" if quant == "4bit" else None,
                bnb_4bit_compute_dtype=torch.float16 if quant == "4bit" else None,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb,
                device_map=device_map,
                max_memory=max_memory,
                offload_folder=offload_dir,
                low_cpu_mem_usage=True,
            )
            return pipeline(
                "text-generation",
                model=model,
                tokenizer=tok,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
            )
        except Exception:
            # bitsandbytes not available or failed -> fall through
            pass

    # fp16 on sufficiently large GPUs
    if (quant in ("auto", "fp16")) and want_cuda and vram_gb >= 10.0:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map=device_map,
            max_memory=max_memory,
            offload_folder=offload_dir,
            low_cpu_mem_usage=True,
        )
        return pipeline(
            "text-generation",
            model=model,
            tokenizer=tok,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
        )

    # CPU fallback (or tiny VRAM or quant failure)
    model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True)
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tok,
        device=-1,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
    )

# ====== main summariser ======
def summarise_text(text: str,
                   word_limit: int = WORD_LIMIT,
                   chunk_size: int = CHUNK_SIZE,
                   overlap: int = CHUNK_OVERLAP) -> str:
    """
    Summarise the transcript via per-chunk summaries + aggregation.
    VRAM-aware model loading with safe CPU retry on OOM.
    """
    try:
        summarizer = _build_textgen_pipeline(
            quant=DEFAULT_QUANT,
            gpu_mem_gb=14,
            max_new_tokens=word_limit * 2
        )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        summarizer = _build_textgen_pipeline(
            quant="auto",
            gpu_mem_gb=0,
            max_new_tokens=word_limit * 2
        )

    # 1) chunk + per-chunk summaries
    chunks = chunk_with_overlap(text, chunk_size, overlap)
    chunk_summaries: List[str] = []

    for idx, chunk in enumerate(chunks):
        prompt = (
            f"Summarise the following meeting transcript in no more than {word_limit} words. "
            "Be concise, factual, and avoid repetition. Highlight the most important points, decisions, and action items.\n\n"
            f"{chunk}\n\nSummary:"
        )
        try:
            out = summarizer(prompt, return_full_text=False)
            summary = out[0]["generated_text"].strip()
            chunk_summaries.append(summary if summary else f"No summary generated for chunk {idx+1}.")
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            summarizer = _build_textgen_pipeline(
                quant="auto",
                gpu_mem_gb=0,
                max_new_tokens=word_limit * 2
            )
            out = summarizer(prompt, return_full_text=False)
            chunk_summaries.append(out[0]["generated_text"].strip())
        except Exception as e:
            chunk_summaries.append(f"Error on chunk {idx+1}: {e}")

    # 2) aggregate
    if len(chunk_summaries) == 1:
        final_summary = chunk_summaries[0]
    else:
        combined = "\n".join(chunk_summaries)
        agg_prompt = (
            f"Combine and rewrite the following draft summaries into a single, cohesive meeting summary of no more than {word_limit} words. "
            "Remove repeated points, ensure logical flow, and avoid splitting sentences.\n\n"
            f"{combined}\n\nFinal summary:"
        )
        out = summarizer(agg_prompt, return_full_text=False)
        final_summary = out[0]["generated_text"].strip()

    return truncate_to_word_limit(final_summary, word_limit)

# ====== CLI ======
if __name__ == "__main__":
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
