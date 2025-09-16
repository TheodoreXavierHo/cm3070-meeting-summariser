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
- Anti-loop + robust canonicalizer: prompts + generator constraints + parser that rebuilds clean Markdown
"""

import os
import re
import sys
from typing import List, Tuple, Iterable, Dict

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# ====== High-level knobs ======
WORD_LIMIT = 500
MAX_INPUT_TOKENS = int(os.getenv("FYP_MAX_INPUT_TOKENS", "2048"))
OVERLAP_TOKENS   = int(os.getenv("FYP_OVERLAP_TOKENS", "256"))
BATCH_SIZE       = int(os.getenv("FYP_BATCH", "2"))
USE_CACHE        = os.getenv("FYP_USE_CACHE", "1") != "0"
REPETITION_PENALTY = float(os.getenv("FYP_REPETITION_PENALTY", "1.15"))
NO_REPEAT_NGRAM  = int(os.getenv("FYP_NO_REPEAT_NGRAM", "8"))

# ====== Granite 3.3 models ======
GRANITE_8B = "ibm-granite/granite-3.3-8b-instruct"
GRANITE_2B = "ibm-granite/granite-3.3-2b-instruct"

# ====== env overrides ======
ENV_MODEL_ID = os.getenv("FYP_MODEL_ID", "").strip()
FORCE_FAMILY = os.getenv("FYP_FORCE_MODEL", "").strip().lower()   # "8b"|"2b"|""(auto)
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
    """Remove common in-line timestamps like [12.34s - 56.78s] at line start."""
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
    return GRANITE_8B

# ====== Model / pipeline builder ======
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

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=REPETITION_PENALTY,
        no_repeat_ngram_size=NO_REPEAT_NGRAM,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )

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
            gen = pipeline("text-generation", model=model, tokenizer=tok, **gen_kwargs)
            gen.model.config.use_cache = USE_CACHE
            return gen, tok, model_id
        except Exception:
            pass

    if (quant in ("auto", "fp16")) and want_cuda and vram_gb >= 10.0:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map=device_map,
            max_memory=max_memory,
            offload_folder=offload_dir,
            low_cpu_mem_usage=True,
        )
        gen = pipeline("text-generation", model=model, tokenizer=tok, **gen_kwargs)
        gen.model.config.use_cache = USE_CACHE
        return gen, tok, model_id

    model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True)
    gen = pipeline("text-generation", model=model, tokenizer=tok, device=-1, **gen_kwargs)
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

# ====== Prompts (markdown-structured) ======
def _mk_chunk_prompt(chunk: str, word_limit: int) -> str:
    """Tight prompt forbidding meta like 'Revised Response' or 'Word Count'."""
    return (
        "You are an expert meeting summariser.\n\n"
        f"Write **ONE** markdown summary of **no more than {word_limit} words total**.\n"
        "Output **GitHub-flavoured Markdown** with these sections **exactly once**, in this order:\n\n"
        "**Meeting Summary**\n\n"
        "**Overview**\n"
        "- One short paragraph.\n\n"
        "**Key Discussion Points**\n"
        "- Bullet points only.\n\n"
        "**Decisions**\n"
        "- Bullet points; write `None` if no explicit decisions.\n\n"
        "**Action Items**\n"
        "1. Numbered list; include **Owner:** and **Deadline:** when present. Write `None` if empty.\n\n"
        "Do not add extra sections or commentary. Do **not** include 'Revised Response', notes, or word/page counts.\n\n"
        "Transcript chunk:\n"
        f"{chunk}\n"
    )

def _mk_agg_prompt(combined: str, word_limit: int) -> str:
    """Aggregation prompt with the same 'no meta' constraints."""
    return (
        f"Combine these draft summaries into **one** cohesive markdown summary of **no more than {word_limit} words**.\n"
        "Return **ONLY** the markdown with these sections **exactly once**, in this order:\n\n"
        "**Meeting Summary**\n\n"
        "**Overview**\n"
        "- One short paragraph.\n\n"
        "**Key Discussion Points**\n"
        "- Bullet points only.\n\n"
        "**Decisions**\n"
        "- Bullet points; write `None` if no explicit decisions.\n\n"
        "**Action Items**\n"
        "1. Numbered list; include **Owner:** and **Deadline:** when present. Write `None` if empty.\n\n"
        "No other headings, footers, or repeated sections. Do **not** include 'Revised Response', notes, or word/page counts.\n\n"
        "Draft summaries:\n"
        f"{combined}\n"
    )

# ====== Canonicalization helpers ======

# canonical headers printed to output (order matters)
_PRINT_HEADERS = [
    "**Meeting Summary**",
    "**Overview**",
    "**Key Discussion Points**",
    "**Decisions**",
    "**Action Items**",
]

# flexible patterns that can match messy model output for each section
_MATCH_PATTERNS: Dict[str, List[str]] = {
    "**Meeting Summary**": [
        r"\*\*Meeting\s+Summary\*\*",
        r"#{1,6}\s*Meeting\s+Summary",
        r"(?:\d+\.\s*)?Meeting\s+Summary",   # numbered heading line
    ],
    "**Overview**": [
        r"\*\*Overview\*\*",
        r"#{1,6}\s*Overview",
        r"(?:\d+\.\s*)?Overview",
    ],
    "**Key Discussion Points**": [
        r"\*\*Key\s*Discussion\s*Points\*\*",
        r"#{1,6}\s*Key\s*Discussion\s*Points",
        r"\*\*Key\s*DiscussionPoints\*\*",
        r"(?:\d+\.\s*)?Key\s*Discussion\s*Points",
    ],
    "**Decisions**": [
        r"\*\*Decisions\*\*",
        r"#{1,6}\s*Decisions?",
        r"(?:\d+\.\s*)?Decisions?",
    ],
    "**Action Items**": [
        r"\*\*Action\s*Items\*\*",
        r"#{1,6}\s*Action\s*Items",
        r"(?:\d+\.\s*)?Action\s*Items",
        r"#{1,6}\s*Action\s*Points",
    ],
}

# lines we should strip entirely (meta/rubric that models tend to add)
_META_LINE = re.compile(
    r"(?im)^\s*(?:word\s*count\s*:.*|note\s*:.*|page\s*limit.*|response complies.*|this response complies.*)\s*$"
)
# a heading that signals a second pass (we cut everything from here)
_REVISED_HDR = re.compile(r"(?im)^\s*(?:#{1,6}\s*)?(?:\d+\.\s*)?revised\s+response\s*$")

# for older “Instruction:” tails
_INSTRUCTION_TAIL = re.compile(r"(?im)^\s*(?:#{1,6}\s*)?Instruction:.*$")

def _strip_meta_and_noise(md: str) -> str:
    """Remove rubric/meta lines and cut at 'Revised Response' or 'Instruction:' tails."""
    # cut at Revised Response section start
    rr = _REVISED_HDR.search(md)
    if rr:
        md = md[: rr.start()]

    # drop anything after an "Instruction:" line
    inst = _INSTRUCTION_TAIL.search(md)
    if inst:
        md = md[: inst.start()]

    # remove typical meta lines like "Word Count:", "Note:", "Page limit…"
    kept = [ln for ln in md.splitlines() if not _META_LINE.match(ln)]
    md = "\n".join(kept)

    # drop explicit "End of Meeting Summary" markers
    md = re.sub(r"\*\*End of Meeting Summary\*\*", "", md, flags=re.IGNORECASE)
    return md.strip()

def _find_first(pats: List[str], text: str) -> int:
    idx = -1
    for p in pats:
        m = re.search(p, text, flags=re.IGNORECASE)
        if m:
            idx = m.start() if idx == -1 else min(idx, m.start())
    return idx

def _extract_sections(md: str) -> Dict[str, str]:
    """
    Parse messy markdown into the five canonical sections.
    Returns a dict mapping printed headers to their content (without the header line).
    """
    md = _strip_meta_and_noise(md)

    # locate each header's first occurrence
    locs = {}
    for hdr, pats in _MATCH_PATTERNS.items():
        pos = _find_first(pats, md)
        if pos != -1:
            locs[hdr] = pos

    if not locs:
        # nothing recognized; treat everything as Overview content
        return {
            "**Meeting Summary**": "",
            "**Overview**": md.strip(),
            "**Key Discussion Points**": "",
            "**Decisions**": "",
            "**Action Items**": "",
        }

    # slice content between headers in the order they appear in the text
    ordered_hits = sorted(locs.items(), key=lambda kv: kv[1])
    slices: Dict[str, str] = {}
    for i, (hdr, start) in enumerate(ordered_hits):
        end = len(md)
        if i + 1 < len(ordered_hits):
            end = ordered_hits[i + 1][1]
        block = md[start:end]

        # remove the header line itself (match one of its patterns)
        header_line_removed = False
        for p in _MATCH_PATTERNS[hdr]:
            m = re.search(p, block, flags=re.IGNORECASE)
            if m:
                block = block[m.end():]
                header_line_removed = True
                break
        if not header_line_removed:
            # best effort: drop first line
            block = "\n".join(block.splitlines()[1:])

        slices[hdr] = block.strip()

    # ensure all five keys present
    for hdr in _PRINT_HEADERS:
        slices.setdefault(hdr, "")

    # normalize content of each section
    slices["**Overview**"] = _clean_overview(slices["**Overview**"])
    slices["**Key Discussion Points**"] = _clean_bullets(slices["**Key Discussion Points**"])
    slices["**Decisions**"] = _clean_bullets(slices["**Decisions**"], allow_none=True)
    slices["**Action Items**"] = _clean_actions(slices["**Action Items**"])

    return slices

def _line_is_any_header(line: str) -> str | None:
    """Return the canonical header key if this line looks like a section heading."""
    s = line.strip()
    for hdr, pats in _MATCH_PATTERNS.items():
        for p in pats:
            # line-level match for bold, ATX, or numbered headings
            if re.match(rf"^\s*(?:{p})\s*$", s, flags=re.IGNORECASE):
                return hdr
    return None

def _keep_single_markdown_block(text: str) -> str:
    """
    Keep only the first pass through our required headers and cut if a second pass starts.
    Detects bold (**Header**), ATX (# Header), numbered ("12. Header"), and setext (Header + =====).
    Also stops if a 'Revised Response' heading appears.
    """
    out_lines: List[str] = []
    seen_counts: Dict[str, int] = {h: 0 for h in _PRINT_HEADERS}
    setext_seen_h1 = False

    for line in text.splitlines():
        s = line.strip()

        # Revised Response → cut here
        if _REVISED_HDR.match(s):
            break

        # setext H1 detection: "Meeting Summary" then "===="
        if re.match(r"(?i)^meeting\s+summary\s*$", s):
            setext_seen_h1 = True
        elif setext_seen_h1 and re.match(r"^=+\s*$", s):
            break  # second pass setext header → cut

        # detect section headers in multiple styles
        hdr = _line_is_any_header(s)
        if hdr is not None:
            seen_counts[hdr] += 1
            if seen_counts[hdr] > 1:
                break  # duplicated heading → cut

        out_lines.append(line)

    return "\n".join(out_lines).strip()

def _clean_overview(txt: str) -> str:
    txt = txt.strip()
    if not txt:
        return "- "
    # collapse multiple lines into a short paragraph bullet
    lines = [l.strip() for l in txt.splitlines() if l.strip()]
    para = " ".join(lines)
    return f"- {para}"

def _clean_bullets(txt: str, allow_none: bool = False) -> str:
    lines = [l.strip() for l in txt.splitlines() if l.strip()]
    bullets = []
    for l in lines:
        # strip leading markers
        l = re.sub(r"^[-*•\d\.\)\(]+\s*", "", l)
        if l.lower() in {"none", "no decisions", "n/a"}:
            if allow_none:
                return "- None"
            else:
                continue
        bullets.append(f"- {l}")
    if not bullets:
        return "- None" if allow_none else "- "
    return "\n".join(bullets)

def _clean_actions(txt: str) -> str:
    lines = [l.rstrip() for l in txt.splitlines() if l.strip()]
    if not lines:
        return "1. None"
    items = []
    for l in lines:
        # turn bullets into numbered items; keep existing numbers
        m = re.match(r"^\s*(\d+)[\.\)]\s*(.+)$", l)
        if m:
            content = m.group(2).strip()
        else:
            content = re.sub(r"^[-*•]\s*", "", l).strip()
        if content.lower() == "none":
            continue
        items.append(content)
    if not items:
        return "1. None"
    return "\n".join(f"{i}. {it}" for i, it in enumerate(items, 1))

def _rebuild_markdown(sections: Dict[str, str]) -> str:
    parts = []
    for hdr in _PRINT_HEADERS:
        parts.append(hdr)
        content = sections.get(hdr, "").strip()
        if content:
            parts.append(content)
        else:
            # sensible defaults
            if hdr == "**Decisions**":
                parts.append("- None")
            elif hdr == "**Action Items**":
                parts.append("1. None")
            else:
                parts.append("- ")
        parts.append("")  # blank line between sections
    return "\n".join(parts).strip()

def _canonicalize_markdown(md: str) -> str:
    """
    Robust normalizer:
    - strip meta lines + cut at 'Revised Response' and other tails
    - keep only first pass of headings (bold/#/numbered/setext)
    - parse sections via flexible regex matches
    - rebuild clean, canonical markdown in the correct order
    """
    md = _keep_single_markdown_block(_strip_meta_and_noise(md))
    secs = _extract_sections(md)
    canon = _rebuild_markdown(secs)
    # de-escape if any stray \** slipped in
    return canon.replace(r"\*\*", "**").strip()

# ====== Main summariser ======
def summarise_text(text: str,
                   word_limit: int = WORD_LIMIT,
                   max_input_tokens: int = MAX_INPUT_TOKENS,
                   overlap_tokens: int = OVERLAP_TOKENS) -> str:
    gen_cap = max(128, int(word_limit * 1.6))

    try:
        summarizer, tok, _ = _build_textgen_pipeline(
            quant=DEFAULT_QUANT, gpu_mem_gb=14, max_new_tokens=gen_cap
        )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        summarizer, tok, _ = _build_textgen_pipeline(
            quant="auto", gpu_mem_gb=0, max_new_tokens=gen_cap
        )

    chunks = _make_token_chunks(tok, text, max_input_tokens, overlap_tokens)
    prompts = [_mk_chunk_prompt(c, word_limit) for c in chunks]

    def batched(iterable: List[str], size: int) -> Iterable[List[str]]:
        for i in range(0, len(iterable), size):
            yield iterable[i:i+size]

    chunk_summaries: List[str] = []
    torch.manual_seed(42)

    for batch in batched(prompts, max(1, BATCH_SIZE)):
        try:
            with torch.inference_mode():
                outs = summarizer(batch, return_full_text=False)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            summarizer, tok, _ = _build_textgen_pipeline(
                quant="auto", gpu_mem_gb=0, max_new_tokens=gen_cap
            )
            with torch.inference_mode():
                outs = summarizer(batch, return_full_text=False)

        for o in outs:
            raw = o[0]["generated_text"].strip()
            chunk_summaries.append(_canonicalize_markdown(raw))

    if len(chunk_summaries) == 1:
        final_md = chunk_summaries[0]
    else:
        combined = "\n\n---\n\n".join(chunk_summaries)
        agg_prompt = _mk_agg_prompt(combined, word_limit)
        with torch.inference_mode():
            out = summarizer(agg_prompt, return_full_text=False)
        final_md = _canonicalize_markdown(out[0]["generated_text"].strip())

    final_md = _truncate_to_word_limit(final_md, word_limit)
    return final_md

def _truncate_to_word_limit(text: str, word_limit: int) -> str:
    """Approximate truncation by words while preserving Markdown lines."""
    words = re.findall(r"\b\w+\b", text)
    if len(words) <= word_limit:
        return text
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
    m = re.search(r"[\.!\?](?:\s|$)", result[::-1])
    if m:
        # cut at the last sentence terminator
        cut = len(result) - m.start()
        result = result[:cut].rstrip()
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
