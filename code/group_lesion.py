"""Group canceller-lesion experiment + four-condition canonical verdict.

Given a partition of the FV head set into ``W``, ``C``, and weak heads,
ablate each subgroup independently and jointly, measuring the readout-token
logit shift relative to the unablated baseline. Apply the four-condition
canonical-cancellation verdict:

  (i)   |Δℓ_W| ≥ 0.10 nats with CI excluding 0
  (ii)  |Δℓ_C| ≥ 0.10 nats with CI excluding 0
  (iii) sign(Δℓ_W) ≠ sign(Δℓ_C)
  (iv)  |Δℓ_both| < |Δℓ_W| + |Δℓ_C|     (joint attenuation)
"""
from __future__ import annotations
import torch
import numpy as np
from typing import Sequence
from .config import LESION_MIN_SHIFT, BOOTSTRAP_B, SHUFFLE_NPERM
from .utils import paired_bootstrap_ci, seed_all


def run(
    model,
    eval_prompts: Sequence[dict],
    partition: dict,
    *,
    strategy: str = "zero",
    seed: int = 0,
) -> dict:
    """Run the W/C/joint group lesion under the chosen ablation strategy."""
    base = _logit_diff_batch(model, eval_prompts)
    W_shift = base - _logit_diff_batch(model, eval_prompts, ablate=partition["writers"], strategy=strategy)
    C_shift = base - _logit_diff_batch(model, eval_prompts, ablate=partition["cancellers"], strategy=strategy)
    both = list(partition["writers"]) + list(partition["cancellers"])
    J_shift = base - _logit_diff_batch(model, eval_prompts, ablate=both, strategy=strategy)

    out = {}
    for name, arr in zip(("W", "C", "both"), (W_shift, C_shift, J_shift)):
        pt, lo, hi = paired_bootstrap_ci(arr, B=BOOTSTRAP_B, seed=seed)
        out[f"{name}_shift"] = pt
        out[f"{name}_ci"] = [lo, hi]
        out[f"{name}_sig"] = (lo > 0) or (hi < 0)
    out["opposite_signs"] = (out["W_shift"] * out["C_shift"]) < 0
    out["attenuated"] = abs(out["both_shift"]) < (abs(out["W_shift"]) + abs(out["C_shift"]))
    out["verdict"] = _verdict(out)
    out["n_writers"] = len(partition["writers"])
    out["n_cancellers"] = len(partition["cancellers"])
    out["strategy"] = strategy
    return out


def _verdict(d: dict) -> str:
    big_W = abs(d["W_shift"]) >= LESION_MIN_SHIFT and d["W_sig"]
    big_C = abs(d["C_shift"]) >= LESION_MIN_SHIFT and d["C_sig"]
    if big_W and big_C and d["opposite_signs"] and d["attenuated"]:
        return "canonical_cancellation_circuit"
    if big_W and not big_C:
        return "writers_dominate"
    if big_C and not big_W:
        return "cancellers_dominate"
    if d["opposite_signs"] and not d["attenuated"]:
        return "opposite_no_attenuation"
    return "co_writers"


def _logit_diff_batch(model, prompts, *, ablate=None, strategy="zero"):
    diffs = np.empty(len(prompts))
    with torch.no_grad():
        for i, p in enumerate(prompts):
            if ablate is None:
                logits = model(p["prompt"], return_type="logits")[0, -1]
            else:
                hooks = []
                for L, H in ablate:
                    def make_hook(H_=H):
                        def fn(act, hook):
                            if strategy == "zero":
                                act[..., H_, :] = 0
                            else:
                                act[..., H_, :] = act[..., H_, :].mean(dim=0, keepdim=True)
                            return act
                        return fn
                    hooks.append((f"blocks.{L}.attn.hook_result", make_hook()))
                logits = model.run_with_hooks(p["prompt"], fwd_hooks=hooks)[0, -1]
            yp = model.tokenizer(" " + p["correct"], add_special_tokens=False)["input_ids"][0]
            yn = model.tokenizer(" " + p["incorrect"], add_special_tokens=False)["input_ids"][0]
            diffs[i] = float(logits[yp] - logits[yn])
    return diffs


# ---------------------------------------------------------------------------
# FV-set sign-shuffle null
# ---------------------------------------------------------------------------
def sign_shuffle_null(
    model,
    eval_prompts: Sequence[dict],
    partition: dict,
    *,
    n_perm: int = SHUFFLE_NPERM,
    seed: int = 12345,
) -> dict:
    """Permute W/C labels within ``W ∪ C`` and recompute |W-C| contrast.

    Returns the empirical two-sided p-value plus null statistics. Used to
    test whether the observed |W_shift − C_shift| contrast can arise from
    arbitrary sign assignments inside the same FV set.
    """
    seed_all(seed)
    rng = np.random.default_rng(seed)
    union = list(partition["writers"]) + list(partition["cancellers"])
    nW = len(partition["writers"])
    obs = run(
        model, eval_prompts,
        partition={"writers": partition["writers"], "cancellers": partition["cancellers"], "weak": []},
        strategy="zero", seed=seed,
    )
    obs_contrast = obs["W_shift"] - obs["C_shift"]

    null = np.empty(n_perm)
    for j in range(n_perm):
        permuted = rng.permutation(union)
        permW, permC = list(permuted[:nW]), list(permuted[nW:])
        d = run(
            model, eval_prompts,
            partition={"writers": permW, "cancellers": permC, "weak": []},
            strategy="zero", seed=seed + j + 1,
        )
        null[j] = d["W_shift"] - d["C_shift"]

    p_emp = max(1.0 / n_perm, float((np.abs(null) >= abs(obs_contrast)).mean()))
    p_gauss = _two_sided_z(obs_contrast, null.mean(), null.std())
    return {
        "n_perm": n_perm,
        "observed_contrast": float(obs_contrast),
        "null_p05": float(np.quantile(null, 0.05)),
        "null_p50": float(np.quantile(null, 0.50)),
        "null_p95": float(np.quantile(null, 0.95)),
        "null_mean": float(null.mean()),
        "null_std":  float(null.std()),
        "p_two_sided": p_emp,
        "p_gauss": p_gauss,
    }


def _two_sided_z(x, mu, sigma):
    from scipy.stats import norm
    if sigma == 0:
        return 0.0 if x == mu else 0.0
    z = (x - mu) / sigma
    return float(2 * norm.sf(abs(z)))
