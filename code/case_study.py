"""L11.H4 single-head case study (Pythia-410M).

Three orthogonal interventions:

  - V-shuffle:        permute V across positions; if DLA collapses, the head
                       reads source content.
  - OV singular:       compute the spectrum of W_O W_V; rank-1 plateau ⇒
                       copy-suppression \\citep{mcdougall2023copy}.
  - V-composition:     ablate the dominant upstream writer L10.H9; if DLA is
                       unchanged, the head is parallel, not V-cascaded.
"""
from __future__ import annotations
import torch
import numpy as np
from typing import Sequence
from . import rule_outs


def run(
    model,
    eval_prompts: Sequence[dict],
    *,
    head: tuple[int, int] = (11, 4),
    upstream_writer: tuple[int, int] = (10, 9),
) -> dict:
    """Run all three interventions on ``head``."""
    return {
        "head": list(head),
        "vshuffle":     rule_outs.v_shuffle(model, head, eval_prompts),
        "ov_spectrum":  _ov_spectrum(model, head),
        "vcomposition": rule_outs.v_cascade(
            model, head, [upstream_writer], eval_prompts,
        ),
    }


def _ov_spectrum(model, head, top_k: int = 25) -> dict:
    L, H = head
    W_O = model.blocks[L].attn.W_O[H]
    W_V = model.blocks[L].attn.W_V[H]
    OV = (W_V @ W_O).float()
    U, S, Vh = torch.linalg.svd(OV, full_matrices=False)
    energy = (S ** 2)
    total = float(energy.sum())
    cum_share = (energy.cumsum(0) / total).cpu().numpy()
    return {
        "frob_norm": float(OV.norm()),
        "top1_share": float(cum_share[0]),
        "top5_share": float(cum_share[4]),
        "top10_share": float(cum_share[9]),
        "cum_share_top_k": cum_share[:top_k].tolist(),
        "singular_values_top_k": S[:top_k].cpu().tolist(),
    }
