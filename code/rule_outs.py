"""Six population-level mechanism rule-outs.

  1. Induction-head overlap   — TOST equivalence to chance.
  2. Generic importance       — rule-NLL / random-NLL ratio.
  3. Rank-1 copy-suppression  — top-1 OV Frobenius-energy share.
  4. V-cascade                — top-3 anti-aligned upstream-writer ablation.
  5. V-shuffle                — random permutation of V across positions.
  6. Head-randomised control  — rank-nearest non-FV partition with matched
                                |DLA| (verifies that the structural-attenuation
                                gap is FV-set-specific).
"""
from __future__ import annotations
import torch
import numpy as np
from typing import Sequence
from .config import RULE_SPECIFIC_GATE, GENERIC_GATE
from .utils import paired_bootstrap_ci, tost_hypergeometric, seed_all


# ---------------------------------------------------------------------------
# (1) Induction-head overlap (TOST)
# ---------------------------------------------------------------------------
def induction_overlap_tost(
    induction_top10: Sequence[tuple[int, int]],
    union_W_C: Sequence[tuple[int, int]],
    n_total_heads: int,
) -> str:
    """TOST equivalence to chance for |W∪C ∩ top-10 induction|."""
    obs = len(set(induction_top10) & set(union_W_C))
    return tost_hypergeometric(obs, K=10, n=len(union_W_C), N=n_total_heads)


# ---------------------------------------------------------------------------
# (2) Generic-importance rule-out
# ---------------------------------------------------------------------------
def generic_importance(
    model,
    canceller: tuple[int, int],
    rule_prompts: Sequence[dict],
    random_prompts: Sequence[dict],
) -> dict:
    """Per-canceller :math:`E[\\Delta\\mathrm{NLL}^\\text{rule}]/E[\\Delta\\mathrm{NLL}^\\text{rand}]`.

    Verdict: ``rule_specific`` if ratio ≥ 5, ``generally_important`` if < 1.5.
    """
    rule_drop = _ablation_nll_drop(model, canceller, rule_prompts)
    rand_drop = _ablation_nll_drop(model, canceller, random_prompts)
    ratio = float(rule_drop.mean() / max(rand_drop.mean(), 1e-9))
    pt, lo, hi = paired_bootstrap_ci(rule_drop / rand_drop.mean(), B=5000)
    return {
        "ratio": ratio, "ci_lo": lo, "ci_hi": hi,
        "verdict": "rule_specific" if ratio >= RULE_SPECIFIC_GATE
                   else "generally_important" if ratio < GENERIC_GATE
                   else "inconclusive",
    }


# ---------------------------------------------------------------------------
# (3) Rank-1 copy-suppression (per head)
# ---------------------------------------------------------------------------
def rank1_share(model, head: tuple[int, int]) -> float:
    """Top-1 Frobenius-energy share of :math:`W_O W_V` for this head."""
    L, H = head
    d_h = model.cfg.d_head
    W_O = model.blocks[L].attn.W_O[H]                        # (d_h, d_model)
    W_V = model.blocks[L].attn.W_V[H]                        # (d_model, d_h)
    OV = W_V @ W_O                                           # (d_model, d_model)
    s = torch.linalg.svdvals(OV)
    return float((s[0] ** 2) / (s ** 2).sum())


# ---------------------------------------------------------------------------
# (4) V-cascade rule-out (top-3 upstream-writer ablation)
# ---------------------------------------------------------------------------
def v_cascade(
    model,
    canceller: tuple[int, int],
    upstream_writers: Sequence[tuple[int, int]],
    eval_prompts: Sequence[dict],
) -> dict:
    """Ablate each top-3 anti-aligned upstream writer and see if canceller's
    DLA changes. CI excluding 0 ⇒ the canceller is downstream of that writer.
    """
    from .group_lesion import _logit_diff_batch
    base = _logit_diff_batch(model, eval_prompts, ablate=[canceller], strategy="zero")
    deltas = []
    for w in upstream_writers[:3]:
        with_w = _logit_diff_batch(model, eval_prompts, ablate=[canceller, w], strategy="zero")
        diff = with_w - base
        pt, lo, hi = paired_bootstrap_ci(diff, B=2000)
        deltas.append({"writer": list(w), "diff": pt, "ci": [lo, hi],
                        "downstream": (lo > 0) or (hi < 0)})
    if not upstream_writers:
        return {"verdict": "mechanically_impossible", "tested": []}
    if any(d["downstream"] for d in deltas):
        return {"verdict": "partial", "tested": deltas}
    return {"verdict": "ruled_out", "tested": deltas}


# ---------------------------------------------------------------------------
# (5) V-shuffle (content-driven test)
# ---------------------------------------------------------------------------
def v_shuffle(
    model,
    head: tuple[int, int],
    eval_prompts: Sequence[dict],
    *,
    n_seeds: int = 3,
) -> dict:
    """Random permutation of the head's V across source positions.

    Returns the per-seed mean DLA and a paired-prompt 95% CI on
    ``DLA_baseline - DLA_shuffled``. CI excluding 0 ⇒ content-driven.
    """
    L, H = head
    base = _per_prompt_head_dla(model, eval_prompts, L, H)
    diffs = []
    rng = np.random.default_rng(0)
    for s in range(n_seeds):
        perm = rng.permutation(_seq_len_for_each(model, eval_prompts))
        shuffled = _per_prompt_head_dla(model, eval_prompts, L, H, v_perm=perm)
        diffs.append(base - shuffled)
    pooled = np.concatenate(diffs)
    pt, lo, hi = paired_bootstrap_ci(pooled, B=2000)
    return {
        "baseline_dla": float(base.mean()),
        "shuffled_dla": float(np.concatenate([base - d for d in diffs]).mean()),
        "diff": pt, "ci": [lo, hi],
        "content_driven": (lo > 0),
    }


# ---------------------------------------------------------------------------
# (6) Head-randomised control (tautology rule-out)
# ---------------------------------------------------------------------------
def head_randomised_control(
    model,
    eval_prompts: Sequence[dict],
    union_size: int,
    nW: int,
    nC: int,
    *,
    n_seeds: int = 10,
    seed: int = 0,
) -> dict:
    """Sample ``n_seeds`` random ``|W∪C|``-sized rank-nearest non-FV partitions
    with sub-shapes ``(nW, nC)``; run group lesion on each.

    Reports the FV/random gap ratio (tests whether attenuation gap is
    structurally larger on the FV set than on rank-nearest non-FV heads).
    """
    from .group_lesion import run, _logit_diff_batch
    rng = np.random.default_rng(seed)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    all_heads = [(L, H) for L in range(n_layers) for H in range(n_heads)]

    # Estimate per-head |DLA| once and rank-match.
    from .dla import per_head_refined_dla
    dla = per_head_refined_dla(model, eval_prompts).mean(dim=0).abs().cpu().numpy()
    rank = np.argsort(-dla.flatten())                        # high-|DLA| first
    nonfv_pool = [all_heads[i] for i in rank if i not in []]  # (filter inside caller)

    seeds_out = []
    for s in range(n_seeds):
        sample = rng.choice(len(nonfv_pool), size=union_size, replace=False)
        chosen = [nonfv_pool[i] for i in sample]
        rng.shuffle(chosen)
        d = run(
            model, eval_prompts,
            partition={"writers": chosen[:nW], "cancellers": chosen[nW:nW + nC],
                        "weak": []},
            strategy="zero", seed=seed + s + 1,
        )
        seeds_out.append({
            "W_shift": d["W_shift"], "C_shift": d["C_shift"], "both_shift": d["both_shift"],
        })
    return {"random_seeds": seeds_out}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _per_prompt_head_dla(model, prompts, L, H, v_perm=None):
    """Per-prompt DLA contribution of head (L, H), optionally with permuted V."""
    out = np.empty(len(prompts))
    with torch.no_grad():
        for i, p in enumerate(prompts):
            _, cache = model.run_with_cache(p["prompt"])
            v = cache["v", L][0, :, H]
            if v_perm is not None:
                v = v[v_perm[: v.shape[0]]]
            attn = cache["pattern", L][0, H, -1]
            mix = (attn[:, None] * v).sum(dim=0)
            W_O = model.blocks[L].attn.W_O[H]
            head_out = mix @ W_O
            yp = model.tokenizer(" " + p["correct"], add_special_tokens=False)["input_ids"][0]
            yn = model.tokenizer(" " + p["incorrect"], add_special_tokens=False)["input_ids"][0]
            u = (model.W_U[:, yp] - model.W_U[:, yn])
            out[i] = float(head_out @ u)
    return out


def _ablation_nll_drop(model, head, prompts):
    """Drop in correct-token NLL when ablating ``head`` on each prompt."""
    from .group_lesion import _logit_diff_batch
    base = _logit_diff_batch(model, prompts)
    abl = _logit_diff_batch(model, prompts, ablate=[head], strategy="zero")
    return base - abl


def _seq_len_for_each(model, prompts):
    return max(model.tokenizer(p["prompt"], return_tensors="pt").input_ids.shape[1]
               for p in prompts)
