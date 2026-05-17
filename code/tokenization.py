"""Per-task token bucketization for QK and per-source DLA.

Each prompt is partitioned into five buckets at tokenizer level:
``BOS``, ``format_prefix``, ``demo_input``, ``demo_label``, ``query_input``.

Bucketization uses character-level offsets returned by HF tokenizers
(`return_offsets_mapping=True`) so the mapping is robust to BPE
artefacts across model families.
"""
from __future__ import annotations
import re
from typing import Iterable


_FORMAT_CHARS = set(", \n:\t")          # purely structural / delimiter characters


def bucketize_tokens(prompt: dict, tokens, tokenizer) -> dict[str, list[int]]:
    """Return a dict ``{bucket: [token_idx, ...]}`` for the given prompt."""
    text = prompt["prompt"]
    # HuggingFace fast tokenisers expose offsets:
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc["offset_mapping"]
    char_classes = _classify_chars(text, prompt)

    out = {b: [] for b in ("BOS", "format_prefix", "demo_input",
                            "demo_label", "query_input")}
    # Models that prepend a BOS token have it at position 0.
    if hasattr(tokenizer, "bos_token_id") and tokens[0].item() == tokenizer.bos_token_id:
        out["BOS"].append(0)
        offsets = [(0, 0)] + offsets

    for tok_idx, (s, e) in enumerate(offsets):
        if s == e == 0 and tok_idx == 0 and not out["BOS"]:
            out["BOS"].append(tok_idx)
            continue
        bucket = _dominant(char_classes[s:e]) if e > s else "format_prefix"
        out[bucket].append(tok_idx)
    return out


def _classify_chars(text: str, prompt: dict) -> list[str]:
    """Return per-character bucket labels for the prompt string."""
    task = prompt.get("task_id", "hierarchical")
    if task in ("hierarchical", "modular"):
        return _classify_chars_rule(text)
    if task in ("antonym", "country-capital"):
        return _classify_chars_vocab(text)
    raise ValueError(f"unknown task_id: {task}")


def _classify_chars_rule(text: str) -> list[str]:
    """Rule-task line format: ``f0,f1,Y\\n`` repeated, query is ``f0q,f1q,``."""
    out = ["format_prefix"] * len(text)
    pos = 0
    for line in text.split("\n"):
        line_end = pos + len(line)
        # demo line: f0,f1,Y    or query: f0q,f1q,
        commas = [m.start() for m in re.finditer(",", line)]
        if len(commas) >= 2:
            # demo_input = chars [0 .. commas[0]) and (commas[0]+1 .. commas[1])
            for i in range(0, commas[0]):
                out[pos + i] = "demo_input"
            for i in range(commas[0] + 1, commas[1]):
                out[pos + i] = "demo_input"
            # demo_label or query_input
            if len(commas) >= 3 or (line_end > pos + commas[1] + 1
                                     and line[commas[1] + 1:].strip().isalpha()):
                # demo: third field is the label
                for i in range(commas[1] + 1, len(line)):
                    out[pos + i] = "demo_label"
            else:
                # query line ending with trailing comma
                for i in range(commas[1] + 1, len(line)):
                    out[pos + i] = "query_input"
        pos = line_end + 1   # +1 for the '\n'
    return out


def _classify_chars_vocab(text: str) -> list[str]:
    """Vocab-task token format: ``a:b a:b ... q:`` (space-separated)."""
    out = ["format_prefix"] * len(text)
    pos = 0
    for token in text.split():
        end = pos + len(token)
        if ":" in token:
            colon = token.index(":")
            for i in range(0, colon):
                out[pos + i] = "demo_input"
            for i in range(colon + 1, len(token)):
                # the trailing space-less query token after the last ':' is empty
                out[pos + i] = "demo_label"
            if end == len(text) or text[end:end + 1] == " " and text[end:].strip() == "":
                for i in range(colon + 1, len(token)):
                    out[pos + i] = "query_input"
        else:
            for i in range(pos, end):
                out[i] = "demo_input"
        pos = end + 1   # +1 for the space
    return out


def _dominant(labels: Iterable[str]) -> str:
    """Pick the most common non-format label, falling back to format."""
    counts: dict[str, int] = {}
    for ch in labels:
        counts[ch] = counts.get(ch, 0) + 1
    counts.pop("format_prefix", None)
    if not counts:
        return "format_prefix"
    return max(counts, key=counts.get)
