"""Path patching with the ratio-of-means estimator.

For each candidate FV head, we ablate its contribution to the residual
stream while holding all downstream paths fixed, and measure the change
in the readout-token logit difference between correct and incorrect
labels. The output is a per-head signed direct effect (positive = head
pushes correct logit up).
"""
from __future__ import annotations
import torch
import numpy as np
from typing import Sequence
from .config import N_PP, PP_DIRECT_GATE


def path_patch_per_head(
    model,
    paired_prompts: Sequence[tuple[dict, dict]],
    fv_heads: Sequence[tuple[int, int]],
    *,
    strategy: str = "zero",
) -> dict:
    """Run path-patching for each head in ``fv_heads``.

    ``paired_prompts`` is a list of ``(correct, rule_flipped)`` pairs.

    Returns
    -------
    {
        "heads": [...],
        "direct_pct": [...],   # signed direct effect normalised by total swing
        "ci_lo": [...],
        "ci_hi": [...],
        "ablation_strategy": "zero" | "mean",
    }
    """
    n = len(paired_prompts)
    swings = np.empty(n)
    direct = {h: np.empty(n) for h in fv_heads}

    with torch.no_grad():
        for i, (xc, xr) in enumerate(paired_prompts):
            base_c = _logit_diff(model, xc)
            base_r = _logit_diff(model, xr)
            swings[i] = base_c - base_r
            for L, H in fv_heads:
                ablated = _logit_diff(model, xc, ablate=(L, H), strategy=strategy)
                direct[(L, H)][i] = ablated - base_r            # remaining swing
    total_swing = swings.mean()
    out = {"heads": list(fv_heads), "direct_pct": [], "ci_lo": [], "ci_hi": [],
           "ablation_strategy": strategy, "total_swing": float(total_swing)}
    from .utils import paired_bootstrap_ci
    for h in fv_heads:
        # head's direct contribution = (full_swing − ablated_swing) / full_swing
        contrib = 1.0 - direct[h] / np.where(swings != 0, swings, 1.0)
        pt, lo, hi = paired_bootstrap_ci(contrib, B=2000)
        out["direct_pct"].append(float(pt))
        out["ci_lo"].append(float(lo))
        out["ci_hi"].append(float(hi))
    return out


def partition_into_W_C_weak(pp_result: dict) -> dict:
    """Apply the ±5% direct-effect gate to split FV heads into W/C/weak."""
    W, C, weak = [], [], []
    for h, p in zip(pp_result["heads"], pp_result["direct_pct"]):
        if p > PP_DIRECT_GATE:
            W.append(h)
        elif p < -PP_DIRECT_GATE:
            C.append(h)
        else:
            weak.append(h)
    return {"writers": W, "cancellers": C, "weak": weak}


# ---------------------------------------------------------------------------
# Helpers (TransformerLens conventions)
# ---------------------------------------------------------------------------
def _logit_diff(model, prompt, *, ablate=None, strategy="zero") -> float:
    if ablate is None:
        with torch.no_grad():
            logits = model(prompt["prompt"], return_type="logits")[0, -1]
    else:
        L, H = ablate
        def hook(activation, hook):
            if strategy == "zero":
                activation[..., H, :] = 0.0
            elif strategy == "mean":
                activation[..., H, :] = activation[..., H, :].mean(dim=0, keepdim=True)
            return activation
        with torch.no_grad():
            logits = model.run_with_hooks(
                prompt["prompt"],
                fwd_hooks=[(f"blocks.{L}.attn.hook_result", hook)],
            )[0, -1]
    yp = model.tokenizer(" " + prompt["correct"], add_special_tokens=False)["input_ids"][0]
    yn = model.tokenizer(" " + prompt["incorrect"], add_special_tokens=False)["input_ids"][0]
    return float(logits[yp] - logits[yn])
