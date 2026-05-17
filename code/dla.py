"""Refined direct logit attribution (DLA) for attention heads.

For each head :math:`(L,H)` and prompt :math:`x`, the refined-DLA is

.. math::

    \\widehat{\\mathrm{DLA}}_{(L,H)}(x) = u(x)^\\top
        \\bigl(\\gamma \\odot
        \\tfrac{c_{(L,H)}(x) - \\overline{c_{(L,H)}}(x)}
              {\\sigma(r(x)) + \\varepsilon}\\bigr)

where

* :math:`c_{(L,H)} = W_O^{(L)}[:, H d_h:(H{+}1) d_h]\\, h_{(L,H)}` is the
  head's contribution to the residual stream after :math:`W_O`,
* :math:`u(x) = W_U[y_+] - W_U[y_-]` is the unembed-difference direction,
* :math:`\\overline{c_{(L,H)}}` is the batch mean over discovery prompts,
* :math:`\\sigma(r(x))` is the standard deviation of the pre-LN residual
  at the last token, and
* :math:`\\gamma` is the final-LN gain (frozen).
"""
from __future__ import annotations
import torch
import numpy as np
from .config import DLA_FDR_Q, DLA_NPERM, N_DISCOVERY, DISCOVERY_SEEDS
from .utils import benjamini_hochberg, seed_all


def per_head_refined_dla(
    model,
    prompts: list[dict],
    *,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Return ``(n_prompts, n_layers, n_heads)`` tensor of refined-DLA values.

    ``model`` must expose ``run_with_cache`` (TransformerLens convention)
    so that we can read pre-LN residual statistics, head outputs, and the
    final LN gain.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    out = torch.empty(len(prompts), n_layers, n_heads)

    # frozen final-LN gain (γ) and unembed
    gamma = model.ln_final.w.detach()
    W_U = model.W_U.detach()

    # discovery-batch reference for the (centred) head output
    with torch.no_grad():
        head_means = _mean_head_output(model, prompts[:N_DISCOVERY])

    for i, p in enumerate(prompts):
        with torch.no_grad():
            _, cache = model.run_with_cache(p["prompt"])
        last = -1
        # Unembed-difference direction
        y_pos = model.tokenizer(" " + p["correct"], add_special_tokens=False)["input_ids"][0]
        y_neg = model.tokenizer(" " + p["incorrect"], add_special_tokens=False)["input_ids"][0]
        u = (W_U[:, y_pos] - W_U[:, y_neg])
        # Pre-LN residual std at last token
        resid = cache["resid_post", n_layers - 1][0, last]
        sigma = resid.std().clamp_min(eps)
        for L in range(n_layers):
            head_out = cache["result", L][0, last]            # (n_heads, d_model)
            centred = head_out - head_means[L]
            scaled = (gamma / sigma) * centred                # broadcast (n_heads, d_model)
            out[i, L] = scaled @ u
    return out


def _mean_head_output(model, prompts):
    n_layers = model.cfg.n_layers
    accum = None
    cnt = 0
    with torch.no_grad():
        for p in prompts:
            _, cache = model.run_with_cache(p["prompt"])
            stack = torch.stack(
                [cache["result", L][0, -1] for L in range(n_layers)], dim=0
            )                                                  # (n_layers, n_heads, d_model)
            if accum is None:
                accum = torch.zeros_like(stack)
            accum += stack
            cnt += 1
    return accum / cnt


# ---------------------------------------------------------------------------
# Permutation null + FV-candidate screening
# ---------------------------------------------------------------------------
def screen_fv_candidates(
    dla: torch.Tensor,
    *,
    q: float = DLA_FDR_Q,
    n_perm: int = DLA_NPERM,
    seed: int = 12345,
    K_min: int = 8,
    K_max: int = 50,
) -> dict:
    """Apply BH-FDR over permutation nulls and return the screened FV set.

    ``dla`` is shape ``(n_prompts, n_layers, n_heads)``. We test the per-head
    mean against an empirical null formed by permuting prompt labels
    ``n_perm`` times.

    Returns ``{"fv_heads": [(L, H), ...], "K": int, "fdr_n": int}``.
    """
    seed_all(seed)
    rng = np.random.default_rng(seed)
    n, n_layers, n_heads = dla.shape
    obs = dla.mean(dim=0).abs().cpu().numpy()                  # (n_layers, n_heads)

    null_max = np.empty(n_perm)
    for j in range(n_perm):
        flips = rng.choice([+1, -1], size=n)
        permed = (dla.cpu().numpy() * flips[:, None, None]).mean(axis=0)
        null_max[j] = np.abs(permed).max()
    p = (null_max[None, None, :] >= obs[..., None]).mean(axis=-1)
    p = p.clip(min=1.0 / n_perm)

    # Restrict to layers L ≥ 1
    p[0] = 1.0
    flat_p = p.flatten()
    sig = benjamini_hochberg(flat_p, q=q).reshape(p.shape)
    n_fdr = int(sig.sum())

    K = int(np.clip(2 * n_fdr, K_min, K_max))
    flat_obs = obs.flatten()
    topk_idx = np.argsort(flat_obs)[-K:][::-1]
    fdr_idx = np.where(flat_p < q)[0]
    keep = np.unique(np.concatenate([topk_idx, fdr_idx]))
    fv_heads = [(int(idx // n_heads), int(idx % n_heads)) for idx in keep]
    return {"fv_heads": sorted(fv_heads), "K": K, "fdr_n": n_fdr}
