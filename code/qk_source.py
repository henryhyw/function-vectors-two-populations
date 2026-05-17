"""QK source-bucket decomposition.

For each W/C head and each prompt, the per-bucket attention mass at the
final query-token position is summed over the canonical buckets:

  - BOS                          : the beginning-of-sequence token.
  - format_prefix                : non-content scaffold tokens
                                   (commas, newlines, separators).
  - demo_input                   : the in-context demo "input" features.
  - demo_label                   : the in-context demo "label" tokens.
  - query_input                  : the final query-input token(s).

The output is the per-cell, per-group mean of these per-head averages.
The load-bearing contrast (canceller − writer) on demo-label and
format-prefix appears as Figure 2 of the paper.
"""
from __future__ import annotations
import torch
import numpy as np
from typing import Sequence

BUCKETS = ("BOS", "format_prefix", "demo_input", "demo_label", "query_input")


def run(
    model,
    eval_prompts: Sequence[dict],
    partition: dict,
) -> dict:
    """Aggregate per-bucket attention mass for the W and C subgroups."""
    n_layers = model.cfg.n_layers
    out = {b: {"writer": 0.0, "canceller": 0.0} for b in BUCKETS}
    counts = {"writer": 0, "canceller": 0}

    with torch.no_grad():
        for p in eval_prompts:
            tokens = model.tokenizer(p["prompt"], return_tensors="pt").input_ids[0]
            bucket_idx = _bucket_token_indices(p, tokens, model.tokenizer)
            _, cache = model.run_with_cache(p["prompt"])
            for group, heads in (
                ("writer", partition["writers"]),
                ("canceller", partition["cancellers"]),
            ):
                for L, H in heads:
                    pattern = cache["pattern", L][0, H, -1]              # (seq,) attention-from-last
                    for b in BUCKETS:
                        idx = bucket_idx.get(b, [])
                        if not idx:
                            continue
                        out[b][group] += float(pattern[idx].sum())
                    counts[group] += 1

    # normalise per head averaged
    n_writers = max(1, counts["writer"])
    n_cancellers = max(1, counts["canceller"])
    return {
        "buckets": list(BUCKETS),
        "n_writers": len(partition["writers"]),
        "n_cancellers": len(partition["cancellers"]),
        "writer_bucket_totals":   {b: out[b]["writer"]    / max(n_writers,   1) for b in BUCKETS},
        "canceller_bucket_totals":{b: out[b]["canceller"] / max(n_cancellers,1) for b in BUCKETS},
    }


def _bucket_token_indices(prompt: dict, tokens, tokenizer) -> dict:
    """Heuristic per-bucket token classifier shared across rule + vocab tasks.

    BOS              : positions of bos token id
    format_prefix    : commas, colons, newlines, leading/trailing whitespace tokens
    demo_input       : contiguous content tokens before each label position
    demo_label       : positions just after the third comma in each demo line
    query_input      : the final-line content tokens before the trailing comma
    """
    # Implementation depends on tokenizer specifics; concrete routine
    # documented in App. A and stable across the eight model families
    # tested in the paper. Released alongside the JSON outputs.
    from .tokenization import bucketize_tokens
    return bucketize_tokens(prompt, tokens, tokenizer)
