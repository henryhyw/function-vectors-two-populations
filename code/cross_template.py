"""Cross-template transfer to vocabulary ICL (antonym, country-capital).

Take the W/C labels learned on a source rule task at one model size and
apply them on a target vocabulary template, then run the relaxed
canonical verdict (drops the joint-attenuation point inequality).

The hier→antonym row is the informative sign-flip: cancellers become
super-writers when the demonstrated label IS the answer. L11.H4 single-head
solo ablation is reported alongside.
"""
from __future__ import annotations
import torch
import numpy as np
from typing import Sequence
from . import prompts as prompt_gen
from . import group_lesion
from .config import N_VOCAB_EVAL


def run(
    model,
    src_partition: dict,
    target_task: str,
    *,
    seed: int = 0,
) -> dict:
    """Apply src-cell W/C labels on target-task prompts; return relaxed verdict."""
    eval_prompts = prompt_gen.generate_batch(target_task, N_VOCAB_EVAL, base_seed=seed)
    out = group_lesion.run(
        model, eval_prompts, src_partition, strategy="zero", seed=seed,
    )
    out["target_task"] = target_task
    out["verdict_relaxed"] = _relaxed_verdict(out)
    return out


def _relaxed_verdict(d: dict) -> str:
    """(i)–(iii) without (iv) attenuation."""
    big_W = abs(d["W_shift"]) >= 0.10 and d["W_sig"]
    big_C = abs(d["C_shift"]) >= 0.10 and d["C_sig"]
    opp = d["opposite_signs"]
    if big_W and big_C and opp:
        return "canonical_relaxed"
    if big_W and not big_C:
        return "writers_dominate"
    if big_C and not big_W:
        return "cancellers_dominate"
    return "no_signal"


# ---------------------------------------------------------------------------
# L11.H4 solo ablation across templates
# ---------------------------------------------------------------------------
def l11h4_solo_per_template(
    model,
    target_tasks: Sequence[str],
    *,
    head: tuple[int, int] = (11, 4),
    seed: int = 0,
) -> dict:
    """Solo-ablate L11.H4 on each target template; report Δℓ + 95% CI."""
    from .group_lesion import _logit_diff_batch
    from .utils import paired_bootstrap_ci

    results = {}
    for task in target_tasks:
        ev = prompt_gen.generate_batch(task, N_VOCAB_EVAL, base_seed=seed)
        base = _logit_diff_batch(model, ev)
        abl = _logit_diff_batch(model, ev, ablate=[head], strategy="zero")
        diff = base - abl
        pt, lo, hi = paired_bootstrap_ci(diff, B=2000, seed=seed)
        results[task] = {
            "mean_shift": pt, "ci": [lo, hi],
            "verdict": ("writer" if hi < 0
                        else "canceller" if lo > 0
                        else "null"),
        }
    return {"head": list(head), "per_task": results}
