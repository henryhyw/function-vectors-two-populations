"""Steering vector construction (sign-aware).

Builds three aggregations of head outputs at the readout token from a
W/C partition:

  - :math:`v_\\text{FV}   = \\overline{\\mathbf{a}}_{\\mathcal{W}\\cup\\mathcal{C}}`   (Todd union mean)
  - :math:`v_\\mathcal{W} = \\overline{\\mathbf{a}}_\\mathcal{W}`                       (drop cancellers)
  - :math:`v_\\text{PCA}  = \\text{top-1 right-singular vector of }
                            [\\overline{\\mathbf{a}}_h]_{h \\in \\mathcal{F}}`,
    sign-aligned to :math:`v_\\mathcal{W}`.

The vectors are inputs to downstream evaluation experiments (held-out
$\\alpha$ injection on the $6$ main cells; transplant accuracy on the
$6$ main cells). Earlier in-sample $\\alpha$ per-vector evaluation
was excluded as confounded (see paper App.~D); this module no longer
provides that evaluation path.
"""
from __future__ import annotations
import torch
from typing import Sequence


def build_steering_vectors(
    model,
    discovery_prompts: Sequence[dict],
    partition: dict,
) -> dict:
    """Compute ``v_FV``, ``v_W``, ``v_PCA`` from per-head mean activations."""
    head_means = _mean_head_outputs(model, discovery_prompts)
    W = partition["writers"]; C = partition["cancellers"]
    union = list(W) + list(C)
    v_FV = torch.stack([head_means[h] for h in union]).mean(dim=0)
    v_W = torch.stack([head_means[h] for h in W]).mean(dim=0) if W else torch.zeros_like(v_FV)
    M = torch.stack([head_means[h] for h in union]).float()
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    v_PCA = Vh[0].to(v_W.dtype)
    if torch.dot(v_PCA, v_W) < 0:
        v_PCA = -v_PCA
    return {"vFV": v_FV, "vW": v_W, "vPCA": v_PCA, "head_means": head_means}


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------
def _mean_head_outputs(model, prompts):
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    accum = {}
    cnt = 0
    with torch.no_grad():
        for p in prompts:
            _, cache = model.run_with_cache(p["prompt"])
            for L in range(n_layers):
                for H in range(n_heads):
                    accum.setdefault((L, H), torch.zeros_like(cache["result", L][0, -1, H]))
                    accum[(L, H)] += cache["result", L][0, -1, H]
            cnt += 1
    return {h: a / cnt for h, a in accum.items()}
