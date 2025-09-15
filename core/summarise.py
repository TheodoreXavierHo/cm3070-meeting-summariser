# core/summarise.py

"""
Summarises a meeting transcript using IBM Granite 3.3 Instruct.

Key features:
- Token-based overlapping chunking (predictable VRAM use)
- Auto-picks 8B vs 2B from VRAM and quant mode (4bit/8bit/fp16/auto)
- Single pipeline build + batched generation for speed
- Optional bitsandbytes quant with GPU offload; safe CPU fallback
- Deterministic, conservative generation settings
- Markdown-structured output for Streamlit (Overview / Key Points / Decisions / Action Items)
"""

import os
import re
import sys
from typing import List, Tuple, Iterable

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# ====== High-level knobs ======
WORD_LIMIT = 500
# Token-based chunking; override via env FYP_MAX_INPUT_TOKENS if needed
MAX_INPUT_TOKENS = int(os.getenv("FYP_MAX_INPUT_TOKENS", "2048"))
OVERLAP_TOKENS   = int(os.getenv("FYP_OVERLAP_TOKENS", "256"))
BATCH_SIZE       = int(os.getenv("FYP_BATCH", "2"))  # prompts per forward pass
USE_CACHE        = os.getenv("FYP_USE_CACHE", "1") != "0"  # KV cache (more VRAM but faster)
REPETITION_PENALTY = float(os.getenv("FYP_REPETITION_PENALTY", "1.05"))

# ====== Granite 3.3 models ======
GRANITE_8B = "ibm-granite/granite-3.3-8b-instruct"
GRANITE_2B = "ibm-granite/granite-3.3-2b-instruct"

# ====== env overrides ======
# FYP_MODEL_ID      -> force a specific HF model id (skips auto-pick)
# FYP_FORCE_MODEL   -> "8b" | "2b" to force family
# FYP_QUANT         -> "auto" | "fp16" | "8bit" | "4bit"   (default: "4bit")
ENV_MODEL_ID = os.getenv("FYP_MODEL_ID", "").strip()
FORCE_FAMILY = os.getenv("FYP_FORCE_MODEL", "").strip().lower()
DEFAULT_QUANT = os.getenv("FYP_QUANT", "4bit").strip().lower()

# ====== Basic I/O ======
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

# ====== Device / VRAM helpers ======
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
    Policy:
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

# ====== Model / pipeline builder ======
def _build_textgen_pipeline(quant: str,
                            gpu_mem_gb: int,
                            max_new_tokens: int):
    """
    Build a HF text-generation pipeline with VRAM-aware config.
    """
    model_id = _choose_family(quant)
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    # Ensure pad/eos are defined for generation; many causal LMs use eos as pad
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    has_cuda, vram_gb = _cuda_vram_gb()
    # allow CUDA from 4 GB upward so 2B-4bit can run on small GPUs
    want_cuda = has_cuda and vram_gb >= 4.0

    device_map = "auto" if want_cuda else None
    # Soft cap VRAM for offload; leave ~2 GB headroom but never below 4 GB
    cap = max(4, int(min(max(vram_gb - 2, 4), gpu_mem_gb))) if want_cuda else 0
    max_memory = {0: f"{cap}GiB", "cpu": "64GiB"} if want_cuda else None

    offload_dir = os.path.join(os.getcwd(), "offload_cache")
    os.makedirs(offload_dir, exist_ok=True)

    # Quantised path first (lowest VRAM)
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
            gen = pipeline(
                "text-generation",
                model=model,
                tokenizer=tok,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                repetition_penalty=REPETITION_PENALTY,
            )
            gen.model.config.use_cache = USE_CACHE
            return gen, tok, model_id
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
        gen = pipeline(
            "text-generation",
            model=model,
            tokenizer=tok,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=REPETITION_PENALTY,
        )
        gen.model.config.use_cache = USE_CACHE
        return gen, tok, model_id

    # CPU fallback (or tiny VRAM / quant failure)
    model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True)
    gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tok,
        device=-1,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=REPETITION_PENALTY,
    )
    gen.model.config.use_cache = USE_CACHE
    return gen, tok, model_id

# ====== Token-based chunking ======
def _make_token_chunks(tokenizer: AutoTokenizer, text: str,
                       max_tokens: int, overlap_tokens: int) -> List[str]:
    """
    Split by tokens (not chars) so each prompt stays within a predictable budget.
    """
    paragraphs = [p for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks: List[str] = []
    cur_ids: List[int] = []
    cur_texts: List[str] = []

    def flush():
        if cur_texts:
            chunks.append("\n\n".join(cur_texts))

    for para in paragraphs:
        ids = tokenizer.encode(para, add_special_tokens=False)
        if len(ids) > max_tokens:
            start = 0
            while start < len(ids):
                end = min(start + max_tokens, len(ids))
                piece_ids = ids[start:end]
                chunks.append(tokenizer.decode(piece_ids))
                start += max_tokens - overlap_tokens
            continue

        if len(cur_ids) + len(ids) > max_tokens:
            flush()
            tail = cur_ids[-overlap_tokens:] if overlap_tokens > 0 else []
            if tail:
                cur_texts = [tokenizer.decode(tail)]
                cur_ids = tail.copy()
            else:
                cur_texts = []
                cur_ids = []

        cur_texts.append(para)
        cur_ids.extend(ids)

    flush()
    return chunks

# ====== Prompts (markdown-structured) ======
def _mk_chunk_prompt(chunk: str, word_limit: int) -> str:
    return (
        f"You are an expert meeting summariser.\n"
        f"Summarise the following transcript content in **no more than {word_limit} words**.\n\n"
        "Output **GitHub-flavoured Markdown** with the EXACT sections and formatting below:\n\n"
        "**Meeting Summary**\n\n"
        "**Overview**\n"
        "- One short paragraph.\n\n"
        "**Key Discussion Points**\n"
        "- Bullet points only.\n\n"
        "**Decisions**\n"
        "- Bullet points; write `None` if no explicit decisions.\n\n"
        "**Action Items**\n"
        "1. Numbered list; include **Owner:** and **Deadline:** when present. Write `None` if empty.\n\n"
        "Do not include the transcript. Do not add extra sections.\n\n"
        "Transcript chunk:\n"
        f"{chunk}\n"
    )

def _mk_agg_prompt(combined: str, word_limit: int) -> str:
    return (
        f"Combine these draft summaries into a single, cohesive meeting summary of **no more than {word_limit} words**.\n\n"
        "Output **GitHub-flavoured Markdown** with the EXACT sections and formatting below:\n\n"
        "**Meeting Summary**\n\n"
        "**Overview**\n"
        "- One short paragraph.\n\n"
        "**Key Discussion Points**\n"
        "- Bullet points only.\n\n"
        "**Decisions**\n"
        "- Bullet points; write `None` if no explicit decisions.\n\n"
        "**Action Items**\n"
        "1. Numbered list; include **Owner:** and **Deadline:** when present. Write `None` if empty.\n\n"
        "Do not add any other sections.\n\n"
        "Draft summaries:\n"
        f"{combined}\n"
    )

# ====== Post-formatting guardrails ======
_SECTION_ORDER = [
    r"\*\*Meeting Summary\*\*",
    r"\*\*Overview\*\*",
    r"\*\*Key Discussion Points\*\*",
    r"\*\*Decisions\*\*",
    r"\*\*Action Items\*\*",
]

def _normalize_markdown(md: str) -> str:
    """Light touch cleanup: ensure required headers exist and spacing is tidy."""
    md = md.strip()

    # Ensure each section header exists at least once
    for header in _SECTION_ORDER:
        if not re.search(rf"(?mi)^{header}\s*$", md):
            # Append missing section at the end with placeholder
            placeholder = "None" if "Decisions" in header or "Action Items" in header else ""
            md += f"\n\n{header}\n"
            if "Overview" in header and not placeholder:
                md += "- \n"
            elif "Key Discussion Points" in header and not placeholder:
                md += "- \n"
            else:
                md += (placeholder or "- ") + "\n"

    # Ensure order by reassembling sections if wildly out of order (best-effort)
    parts = {}
    for i, header in enumerate(_SECTION_ORDER):
        m = re.search(rf"(?mi)^{header}\s*$", md)
        if m:
            start = m.start()
            # find next header
            next_pos = len(md)
            for j in range(i + 1, len(_SECTION_ORDER)):
                n = re.search(rf"(?mi)^{_SECTION_ORDER[j]}\s*$", md)
                if n:
                    next_pos = min(next_pos, n.start())
            parts[header] = md[m.start():next_pos].strip()

    if len(parts) >= 3:
        md = "\n\n".join([parts[h] for h in _SECTION_ORDER if h in parts]).strip()

    # Collapse excess blank lines
    md = re.sub(r"\n{3,}", "\n\n", md).strip()
    return md

# ====== Main summariser ======
def summarise_text(text: str,
                   word_limit: int = WORD_LIMIT,
                   max_input_tokens: int = MAX_INPUT_TOKENS,
                   overlap_tokens: int = OVERLAP_TOKENS) -> str:
    """
    Summarise via per-chunk summaries (batched) + aggregation.
    VRAM-aware model loading with safe CPU retry on OOM.
    """
    try:
        summarizer, tok, model_id = _build_textgen_pipeline(
            quant=DEFAULT_QUANT,
            gpu_mem_gb=14,
            max_new_tokens=word_limit * 2
        )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        summarizer, tok, model_id = _build_textgen_pipeline(
            quant="auto",
            gpu_mem_gb=0,
            max_new_tokens=word_limit * 2
        )

    # Build token-based chunks using the same tokenizer we’ll use for generation
    chunks = _make_token_chunks(tok, text, max_input_tokens, overlap_tokens)

    # Per-chunk prompts
    prompts = [_mk_chunk_prompt(c, word_limit) for c in chunks]

    # Batched generation for speed
    def batched(iterable: List[str], size: int) -> Iterable[List[str]]:
        for i in range(0, len(iterable), size):
            yield iterable[i:i+size]

    chunk_summaries: List[str] = []
    torch.manual_seed(42)  # mild determinism

    for batch in batched(prompts, max(1, BATCH_SIZE)):
        try:
            with torch.inference_mode():
                outs = summarizer(batch, return_full_text=False)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            # Retry whole batch on CPU
            summarizer, tok, _ = _build_textgen_pipeline(
                quant="auto", gpu_mem_gb=0, max_new_tokens=word_limit * 2
            )
            with torch.inference_mode():
                outs = summarizer(batch, return_full_text=False)
        for o in outs:
            chunk_summaries.append(_normalize_markdown(o[0]["generated_text"].strip()))

    # Aggregate (single pass)
    if len(chunk_summaries) == 1:
        final_md = chunk_summaries[0]
    else:
        combined = "\n\n---\n\n".join(chunk_summaries)
        agg_prompt = _mk_agg_prompt(combined, word_limit)
        with torch.inference_mode():
            out = summarizer(agg_prompt, return_full_text=False)
        final_md = _normalize_markdown(out[0]["generated_text"].strip())

    # Hard cap by words without cutting sentences (on raw text)
    final_md = _truncate_to_word_limit(final_md, word_limit)
    return final_md

def _truncate_to_word_limit(text: str, word_limit: int) -> str:
    words = re.findall(r"\b\w+\b", text)
    if len(words) <= word_limit:
        return text
    # Approximate cut by words while preserving markdown lines
    tokens = re.split(r"(\b\w+\b)", text)
    count = 0
    out = []
    for t in tokens:
        if re.fullmatch(r"\b\w+\b", t):
            count += 1
        out.append(t)
        if count >= word_limit:
            break
    result = "".join(out).rstrip()
    # Try to end at a sentence or line break
    m = re.search(r"[\.!\?]\s+[^\n]*$", result)
    if m:
        result = result[:m.end()].rstrip()
    return result

# ====== CLI ======
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python summarise.py <transcript.txt>")
        sys.exit(1)
    transcript_path = sys.argv[1]
    summary_path = "outputs/summary.txt"  # .txt on purpose; contains Markdown

    text = load_transcript(transcript_path)
    text = strip_timestamps(text)

    summary = summarise_text(text)
    save_summary(summary, summary_path)
    print(f"Summary saved to {summary_path}")
