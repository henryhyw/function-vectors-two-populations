"""Per-source DLA bucket attribution for canceller heads.

For each canceller :math:`h`, we attribute its negative DLA to the five
prompt-source buckets by writing the head output as
:math:`\\sum_\\text{pos} \\alpha_\\text{pos} W_O W_V r_\\text{pos}` and
projecting each bucket-sum through the frozen final-LN gain onto
:math:`u(x) = W_U[y_+] - W_U[y_-]`.

A head is labelled CONTENT if the attribution from non-sink buckets
(``demo_input + demo_label + query_input``) dominates; SINK otherwise.
"""
from __future__ import annotations
import torch
import numpy as np
from typing import Sequence

CONTENT_BUCKETS = ("demo_input", "demo_label", "query_input")
SINK_BUCKETS = ("BOS", "format_prefix")


def run(
    model,
    eval_prompts: Sequence[dict],
    cancellers: Sequence[tuple[int, int]],
) -> dict:
    """Decompose each canceller's mean DLA into per-bucket contributions."""
    per_head = {}
    for L, H in cancellers:
        per_head[(L, H)] = _per_bucket_dla(model, eval_prompts, L, H)

    summary = {"cancellers": {}, "content": {}, "sink": {}}
    for h, b in per_head.items():
        summary["cancellers"][f"L{h[0]}.H{h[1]}"] = {
            "total_dla_mean": b["total"],
            "bucket_dla_mean": b["per_bucket"],
        }
    # Per-cell content/sink classification + sums
    label = {h: _classify(b) for h, b in per_head.items()}
    for h, b in per_head.items():
        bucket = "content" if label[h] == "content" else "sink"
        key = f"L{h[0]}.H{h[1]}"
        summary[bucket][key] = b["total"]
    summary["counts"] = {
        "n_content": sum(1 for v in label.values() if v == "content"),
        "n_sink": sum(1 for v in label.values() if v == "sink"),
    }
    return summary


def _per_bucket_dla(model, prompts, L, H, eps: float = 1e-5) -> dict:
    """Average bucket-decomposed DLA for head (L, H) across prompts."""
    from .tokenization import bucketize_tokens
    n_layers = model.cfg.n_layers
    gamma = model.ln_final.w.detach()
    W_U = model.W_U.detach()
    bucket_sum = {b: 0.0 for b in ("BOS", "format_prefix",
                                    "demo_input", "demo_label", "query_input")}
    total = 0.0
    with torch.no_grad():
        for p in prompts:
            tokens = model.tokenizer(p["prompt"], return_tensors="pt").input_ids[0]
            buckets = bucketize_tokens(p, tokens, model.tokenizer)
            _, cache = model.run_with_cache(p["prompt"])
            yp = model.tokenizer(" " + p["correct"], add_special_tokens=False)["input_ids"][0]
            yn = model.tokenizer(" " + p["incorrect"], add_special_tokens=False)["input_ids"][0]
            u = W_U[:, yp] - W_U[:, yn]
            resid = cache["resid_post", n_layers - 1][0, -1]
            sigma = resid.std().clamp_min(eps)

            attn = cache["pattern", L][0, H, -1]                     # (seq,)
            v_proj = cache["v", L][0, :, H]                          # (seq, d_h)
            W_O_head = model.blocks[L].attn.W_O[H]                   # (d_h, d_model)
            head_per_pos = attn[:, None] * (v_proj @ W_O_head)       # (seq, d_model)
            head_per_pos = (gamma / sigma) * head_per_pos
            dla_per_pos = head_per_pos @ u                           # (seq,)

            for b in bucket_sum:
                idx = buckets.get(b, [])
                if idx:
                    bucket_sum[b] += float(dla_per_pos[idx].sum())
            total += float(dla_per_pos.sum())
    n = len(prompts)
    return {
        "total": total / n,
        "per_bucket": {b: bucket_sum[b] / n for b in bucket_sum},
    }


def _classify(b: dict) -> str:
    sink = abs(sum(b["per_bucket"][k] for k in SINK_BUCKETS))
    content = abs(sum(b["per_bucket"][k] for k in CONTENT_BUCKETS))
    return "sink" if sink > content else "content"
