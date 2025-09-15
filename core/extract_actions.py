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
- Output format (strict, single line per item):
1. **Task** — **Owner:** <Name or 'Unassigned'> — **Deadline:** <Date or 'None'>
"""

import os
import re
import sys
import torch
from typing import List, Tuple, Iterable, Dict, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# -------------------- Tunables / env --------------------

# ====== High-level knobs ======
MAX_INPUT_TOKENS      = int(os.getenv("FYP_MAX_INPUT_TOKENS", "2048"))
OVERLAP_TOKENS        = int(os.getenv("FYP_OVERLAP_TOKENS", "256"))
BATCH_SIZE            = int(os.getenv("FYP_BATCH", "2"))
USE_CACHE             = os.getenv("FYP_USE_CACHE", "1") != "0"
REPETITION_PENALTY    = float(os.getenv("FYP_REPETITION_PENALTY", "1.15"))
MAX_NEW_TOKENS_ACTION = int(os.getenv("FYP_MAX_NEW_TOKENS_ACTION", "384"))

# ====== Granite 3.3 models ======
GRANITE_8B = "ibm-granite/granite-3.3-8b-instruct"
GRANITE_2B = "ibm-granite/granite-3.3-2b-instruct"

# ====== env overrides ======
ENV_MODEL_ID  = os.getenv("FYP_MODEL_ID", "").strip()
FORCE_FAMILY  = os.getenv("FYP_FORCE_MODEL", "").strip().lower()   # "8b"|"2b"|""(auto)
DEFAULT_QUANT = os.getenv("FYP_QUANT", "4bit").strip().lower()

# -------------------- IO helpers --------------------

def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def save_text(text: str, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text.strip() + "\n")

def strip_timestamps(text: str) -> str:
    return re.sub(r"^\[\d+\.\d+s\s*-\s*\d+\.\d+s\]\s*", "", text, flags=re.MULTILINE)

# -------------------- Device / model --------------------

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

    has_cuda, vram = _cuda_vram_gb()
    if not has_cuda:
        return GRANITE_2B
    if vram <= 6.0:
        return GRANITE_2B
    if 7.0 <= vram < 12.0:
        if quant == "4bit" and vram >= 8.0:
            return GRANITE_8B
        return GRANITE_2B
    return GRANITE_8B

def _build_textgen_pipeline(quant: str, max_new_tokens: int):
    model_id = _choose_family(quant)
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    has_cuda, vram = _cuda_vram_gb()
    want_cuda = has_cuda and vram >= 4.0

    device_map = "auto" if want_cuda else None
    max_memory = {0: f"{max(4, int(vram) - 2)}GiB", "cpu": "64GiB"} if want_cuda else None
    offload_dir = os.path.join(os.getcwd(), "offload_cache")
    os.makedirs(offload_dir, exist_ok=True)

    # Quantised
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
            return gen, tok
        except Exception:
            pass

    # fp16 if roomy
    if (quant in ("auto", "fp16")) and want_cuda and vram >= 10.0:
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
        return gen, tok

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
    return gen, tok

# -------------------- Chunking --------------------

# ====== Token-based chunking ======
def _make_token_chunks(tokenizer: AutoTokenizer, text: str,
                       max_tokens: int, overlap_tokens: int) -> List[str]:
    paras = [p for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks: List[str] = []
    cur_ids: List[int] = []
    cur_texts: List[str] = []

    def flush():
        if cur_texts:
            chunks.append("\n\n".join(cur_texts))

    for para in paras:
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
            cur_texts = [tokenizer.decode(tail)] if tail else []
            cur_ids = tail.copy() if tail else []

        cur_texts.append(para)
        cur_ids.extend(ids)

    flush()
    return chunks

# -------------------- Prompts --------------------

def _mk_chunk_prompt(chunk: str) -> str:
    return (
        "You are extracting concrete action items from a meeting transcript.\n"
        "Return ONLY a numbered list; no prose, no headings.\n"
        "Use EXACTLY this one-line template per item:\n"
        "1. **<Task>** — **Owner:** <Name or 'Unassigned'> — **Deadline:** <Date or 'None'>\n\n"
        "Ignore: navigation links like [Link 1], and markers like 'End of Transcript/Meeting'.\n\n"
        "Transcript chunk:\n"
        f"{chunk}\n"
    )

def _mk_merge_prompt(lists: str) -> str:
    return (
        "Merge and deduplicate the following numbered lists of action items.\n"
        "Return ONLY a single numbered list; use this template per item:\n"
        "1. **<Task>** — **Owner:** <Name or 'Unassigned'> — **Deadline:** <Date or 'None'>\n"
        "Ignore link blobs and transcript boundary markers.\n\n"
        f"{lists}\n"
    )

# -------------------- Parsing & normalisation --------------------

NOISE_PATTERNS = [
    r'\bend of transcript\b', r'\bstart of transcript\b', r'\bend of meeting\b',
    r'^\s*(?:[-–—_]+)\s*$'
]

def _split_numbered_blocks(md: str) -> List[str]:
    """Return list of blocks, each the text after 'N.' up to the next number."""
    md = md.strip()
    if not md:
        return []
    blocks: List[str] = []
    iters = list(re.finditer(r'^\s*\d+\.\s+', md, flags=re.MULTILINE))
    for i, m in enumerate(iters):
        start = m.end()
        end = iters[i + 1].start() if i + 1 < len(iters) else len(md)
        block = md[start:end].strip()
        if block:
            blocks.append(block)
    return blocks

def _clean(text: str) -> str:
    # drop [link] tokens but keep surrounding text
    text = re.sub(r'\[[^\]]+\]', '', text)
    # collapse whitespace/dashes
    text = re.sub(r'\s*—\s*—\s*', ' — ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()

def _extract_fields(block: str) -> Optional[Dict[str, str]]:
    b = _clean(block)
    low = b.lower()
    if any(re.search(p, low) for p in NOISE_PATTERNS):
        return None

    # Task: first bold section or text before Owner:
    m_task = re.search(r'\*\*(.+?)\*\*', b)
    task = m_task.group(1).strip() if m_task else re.split(r'\s*—\s*\**Owner\**\s*:?', b)[0].strip()
    task = re.sub(r'^\W+|\W+$', '', task)
    if len(re.sub(r'[^a-zA-Z]+', '', task)) < 3:  # must have some letters
        return None

    # Owner
    m_owner = re.search(r'Owner\**\s*:\s*([^\n—*]+)', b, flags=re.IGNORECASE)
    owner = m_owner.group(1).strip() if m_owner else "Unassigned"

    # Deadline
    m_dead = re.search(r'Deadline\**\s*:\s*([^\n—*]+)', b, flags=re.IGNORECASE)
    deadline = m_dead.group(1).strip() if m_dead else "None"

    return {"Task": task, "Owner": owner, "Deadline": deadline}

def _format_items(items: List[Dict[str, str]]) -> str:
    out = []
    for i, it in enumerate(items, 1):
        task = it["Task"].strip()
        owner = it.get("Owner", "Unassigned").strip() or "Unassigned"
        dead  = it.get("Deadline", "None").strip() or "None"
        out.append(f"{i}. **{task}** — **Owner:** {owner} — **Deadline:** {dead}")
    return "\n".join(out) if out else "1. **None** — **Owner:** Unassigned — **Deadline:** None"

def _parse_model_output(md: str) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    seen = set()
    for block in _split_numbered_blocks(md):
        fields = _extract_fields(block)
        if not fields:
            continue
        key = fields["Task"].lower()
        if key in seen:
            continue
        seen.add(key)
        items.append(fields)
    return items

# -------------------- Core extraction --------------------

def extract_action_items(text: str) -> str:
    # Build generator
    try:
        generator, tok = _build_textgen_pipeline(DEFAULT_QUANT, MAX_NEW_TOKENS_ACTION)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        generator, tok = _build_textgen_pipeline("auto", MAX_NEW_TOKENS_ACTION)

    # Chunk transcript
    chunks = _make_token_chunks(tok, text, MAX_INPUT_TOKENS, OVERLAP_TOKENS)
    prompts = [_mk_chunk_prompt(c) for c in chunks]

    def batched(seq: List[str], n: int) -> Iterable[List[str]]:
        for i in range(0, len(seq), max(1, n)):
            yield seq[i:i+max(1, n)]

    partial_md: List[str] = []
    torch.manual_seed(42)

    # Generate per-chunk lists
    for batch in batched(prompts, BATCH_SIZE):
        try:
            with torch.inference_mode():
                outs = generator(batch, return_full_text=False)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            with torch.inference_mode():
                outs = generator(batch, return_full_text=False)
        for o in outs:
            partial_md.append(o[0]["generated_text"].strip())

    # If nothing came back, fail closed with "None"
    if not any(s.strip() for s in partial_md):
        return "1. **None** — **Owner:** Unassigned — **Deadline:** None"

    # Merge (LLM) when multiple chunks
    if len(partial_md) == 1:
        merged = partial_md[0]
    else:
        merged_in = "\n\n---\n\n".join(partial_md)
        with torch.inference_mode():
            out = generator(_mk_merge_prompt(merged_in), return_full_text=False)
        merged = out[0]["generated_text"].strip()

    # Parse model output by blocks; rebuild clean list
    items = _parse_model_output(merged)

    # Fallback: parse the unmerged partials if merged got over-filtered
    if not items:
        for md in partial_md:
            items.extend(_parse_model_output(md))

        # Deduplicate after fallback
        dedup, seen = [], set()
        for it in items:
            k = it["Task"].lower()
            if k in seen: continue
            seen.add(k); dedup.append(it)
        items = dedup

    return _format_items(items)

# -------------------- CLI --------------------

if __name__ == "__main__":
    # Usage: python extract_actions.py <transcript.txt>
    if len(sys.argv) < 2:
        print("Usage: python core/extract_actions.py <combined_transcript.txt>")
        sys.exit(1)

    src_path = sys.argv[1]
    out_path = "outputs/action_items.txt"

    raw = strip_timestamps(load_text(src_path))
    md = extract_action_items(raw)
    save_text(md, out_path)
    print(f"Action items saved to {out_path}")
