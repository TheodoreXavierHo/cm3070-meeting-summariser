# core/extract_actions.py
"""
Extracts action items from a meeting transcript using IBM Granite 3.3 Instruct.

Key features:
- Token-based overlapping chunking (predictable VRAM use)
- Auto-picks 8B vs 2B from VRAM and quant mode (4bit/8bit/fp16/auto)
- Single pipeline build + batched generation for speed
- Optional bitsandbytes quant with GPU offload; safe CPU fallback
- Deterministic generation
- Markdown formatted exactly for Streamlit parser:
  1. **Task** — **Owner:** <Name or 'Unassigned'> — **Deadline:** <Date or 'None'>
"""

import os
import re
import sys
from typing import List, Tuple, Iterable

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# ====== High-level knobs ======
MAX_INPUT_TOKENS      = int(os.getenv("FYP_MAX_INPUT_TOKENS", "2048"))
OVERLAP_TOKENS        = int(os.getenv("FYP_OVERLAP_TOKENS", "256"))
BATCH_SIZE            = int(os.getenv("FYP_BATCH", "2"))
USE_CACHE             = os.getenv("FYP_USE_CACHE", "1") != "0"
REPETITION_PENALTY    = float(os.getenv("FYP_REPETITION_PENALTY", "1.05"))
MAX_NEW_TOKENS_ACTION = int(os.getenv("FYP_MAX_NEW_TOKENS_ACTION", "256"))

# ====== Granite 3.3 models ======
GRANITE_8B = "ibm-granite/granite-3.3-8b-instruct"
GRANITE_2B = "ibm-granite/granite-3.3-2b-instruct"

# ====== env overrides ======
ENV_MODEL_ID  = os.getenv("FYP_MODEL_ID", "").strip()
FORCE_FAMILY  = os.getenv("FYP_FORCE_MODEL", "").strip().lower()   # "8b"|"2b"|""(auto)
DEFAULT_QUANT = os.getenv("FYP_QUANT", "4bit").strip().lower()

# ====== Basic I/O ======
def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def save_actions(actions_md: str, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(actions_md.strip())

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
    return GRANITE_8B  # vram >= 12

# ====== Pipeline builder ======
def _build_textgen_pipeline(quant: str,
                            gpu_mem_gb: int,
                            max_new_tokens: int):
    model_id = _choose_family(quant)
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    has_cuda, vram_gb = _cuda_vram_gb()
    want_cuda = has_cuda and vram_gb >= 4.0  # allow 2B-4bit on small GPUs

    device_map = "auto" if want_cuda else None
    cap = max(4, int(min(max(vram_gb - 2, 4), gpu_mem_gb))) if want_cuda else 0
    max_memory = {0: f"{cap}GiB", "cpu": "64GiB"} if want_cuda else None
    offload_dir = os.path.join(os.getcwd(), "offload_cache")
    os.makedirs(offload_dir, exist_ok=True)

    # Quantised path
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
            pass

    # fp16 path
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

    # CPU fallback
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

# ====== Prompts (strict markdown to match Streamlit) ======
def _mk_chunk_prompt(chunk: str) -> str:
    return (
        "You are extracting concrete action items from a meeting transcript.\n\n"
        "Output **ONLY** a numbered list in GitHub-flavoured Markdown with EXACTLY this template per item:\n"
        "1. **<Task>** — **Owner:** <Name or 'Unassigned'> — **Deadline:** <Date or 'None'>\n\n"
        "Rules:\n"
        "- Include only actionable tasks that were agreed or clearly implied.\n"
        "- If owner/deadline is unknown, write 'Unassigned' / 'None'.\n"
        "- Do NOT add extra sections, commentary, or headings.\n\n"
        "Transcript chunk:\n"
        f"{chunk}\n"
    )

def _mk_agg_prompt(combined_md_lists: str) -> str:
    return (
        "Merge and deduplicate the following numbered lists of action items.\n"
        "Output a SINGLE numbered list in GitHub-flavoured Markdown using EXACTLY this template per item:\n"
        "1. **<Task>** — **Owner:** <Name or 'Unassigned'> — **Deadline:** <Date or 'None'>\n\n"
        "Rules:\n"
        "- Merge duplicates (same/similar task) and keep one precise version.\n"
        "- Preserve owners and deadlines when available.\n"
        "- No headings, no commentary, only the numbered list.\n\n"
        "Lists to merge:\n"
        f"{combined_md_lists}\n"
    )

# ====== Simple deduper shaped for the Streamlit parser ======
def _normalize_and_dedupe_numbered(md_list: str) -> str:
    lines = [l.rstrip() for l in md_list.strip().splitlines() if l.strip()]
    items = []
    seen_keys = set()

    def canon_task(line: str) -> str:
        # Extract the bolded **Task** if present; otherwise the line up to Owner
        m = re.search(r"\*\*(.+?)\*\*", line)
        task = m.group(1) if m else re.sub(r"\s*—\s*\*\*Owner:.*$", "", line)
        task = re.sub(r"\s+", " ", task).strip().lower()
        return task

    for ln in lines:
        # Force numbered prefix
        ln = re.sub(r"^\s*(?:[-•]|\d+\.)\s*", "", ln)
        # Ensure bold task exists
        if "**" not in ln:
            # Try to bold leading phrase before Owner/Deadline
            head = re.split(r"\s*—\s*\*\*Owner:", ln)[0].strip()
            if head:
                ln = f"**{head}** — " + re.sub(r"^\s*"+re.escape(head)+r"\s*(—\s*)?", "", ln)
        # Ensure Owner/Deadline fields
        if "**Owner:**" not in ln:
            ln = f"{ln} — **Owner:** Unassigned"
        if "**Deadline:**" not in ln:
            ln = f"{ln} — **Deadline:** None"

        key = canon_task(ln)
        if key not in seen_keys:
            seen_keys.add(key)
            items.append(ln)

    # Renumber
    out = []
    for i, it in enumerate(items, 1):
        it = re.sub(r"^\s*(?:\d+\.\s*)?", f"{i}. ", it)
        out.append(it)
    return "\n".join(out) if out else "1. **None** — **Owner:** Unassigned — **Deadline:** None"

# ====== Main extractor ======
def extract_action_items(text: str,
                         max_input_tokens: int = MAX_INPUT_TOKENS,
                         overlap_tokens: int = OVERLAP_TOKENS) -> str:
    """
    Extract action items by per-chunk lists (batched) then LLM merge/dedupe.
    """
    try:
        generator, tok, _ = _build_textgen_pipeline(
            quant=DEFAULT_QUANT,
            gpu_mem_gb=14,
            max_new_tokens=MAX_NEW_TOKENS_ACTION
        )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        generator, tok, _ = _build_textgen_pipeline(
            quant="auto",
            gpu_mem_gb=0,
            max_new_tokens=MAX_NEW_TOKENS_ACTION
        )

    # Token chunks
    chunks = _make_token_chunks(tok, text, max_input_tokens, overlap_tokens)
    prompts = [_mk_chunk_prompt(c) for c in chunks]

    def batched(iterable: List[str], size: int) -> Iterable[List[str]]:
        for i in range(0, len(iterable), size):
            yield iterable[i:i+size]

    partial_lists: List[str] = []
    torch.manual_seed(42)

    for batch in batched(prompts, max(1, BATCH_SIZE)):
        try:
            with torch.inference_mode():
                outs = generator(batch, return_full_text=False)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            generator, tok, _ = _build_textgen_pipeline(
                quant="auto", gpu_mem_gb=0, max_new_tokens=MAX_NEW_TOKENS_ACTION
            )
            with torch.inference_mode():
                outs = generator(batch, return_full_text=False)
        for o in outs:
            partial_lists.append(o[0]["generated_text"].strip())

    # Merge + dedupe with LLM once
    if len(partial_lists) == 1:
        merged = partial_lists[0]
    else:
        merged_input = "\n\n---\n\n".join(partial_lists)
        agg_prompt = _mk_agg_prompt(merged_input)
        with torch.inference_mode():
            out = generator(agg_prompt, return_full_text=False)
        merged = out[0]["generated_text"].strip()

    # Final tidy + dedupe; enforce numbered + bold task + Owner/Deadline
    final_md = _normalize_and_dedupe_numbered(merged)
    return final_md

# ====== CLI ======
if __name__ == "__main__":
    # Usage: python extract_actions.py <transcript.txt>
    if len(sys.argv) < 2:
        print("Usage: python extract_actions.py <transcript.txt>")
        sys.exit(1)
    transcript_path = sys.argv[1]
    actions_path = "outputs/action_items.txt"  # .txt on purpose; contains Markdown

    raw = load_text(transcript_path)
    raw = strip_timestamps(raw)
    actions_md = extract_action_items(raw)
    save_actions(actions_md, actions_path)
    print(f"Action items saved to {actions_path}")
