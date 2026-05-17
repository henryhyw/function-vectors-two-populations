"""Cross-cell aggregation: verdict matrix, FWER correction, headline numbers.

Reads every per-cell JSON written by the pipeline and produces the
summaries consumed by ``draft/figures/`` and the manuscript tables:

  - ``results/{task}/_aggregate/cross_cell_fdr.json``
       BH-FDR + Holm-Bonferroni across the 6-cell sign-shuffle family.
  - ``results/_aux/<theme>/_summary.json``
       per-theme aggregate (e.g. mean rel-collapse for V-shuffle).
  - ``../extracted_numbers.json``
       every numerical claim referenced from the LaTeX, keyed by claim
       location, regenerated from the JSONs.
"""
from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from . import config, io
from .utils import benjamini_hochberg, holm_bonferroni


def write_all() -> None:
    """Run all aggregation steps in order. Idempotent."""
    write_sign_shuffle_fdr()
    write_verdict_matrix()
    write_extracted_numbers()


# ---------------------------------------------------------------------------
# Sign-shuffle FDR (BH-q at 0.10 + Holm-Bonferroni at α=0.05)
# ---------------------------------------------------------------------------
def write_sign_shuffle_fdr() -> None:
    rows = []
    for task in ("hierarchical", "modular"):
        for spec in config.PYTHIA_MAIN:
            d = io.aux_load("sign_shuffle_n10k", f"sign_shuffle_{task}_{spec.short}")
            rows.append({
                "task": task, "model": spec.short,
                "p_emp": d["p_two_sided"], "p_gauss": d["p_gauss"],
                "obs": d["observed_contrast"],
            })
    p_emp = [r["p_emp"] for r in rows]
    bh = benjamini_hochberg(p_emp, q=0.10)
    holm = holm_bonferroni(p_emp, alpha=0.05)
    for r, b, h in zip(rows, bh, holm):
        r["bh_q"] = bool(b)
        r["holm_05"] = bool(h)

    for task in ("hierarchical", "modular"):
        sub = [r for r in rows if r["task"] == task]
        out = config.RESULTS_DIR / task / "_aggregate"
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "cross_cell_fdr.json", "w") as f:
            json.dump(sub, f, indent=2)


# ---------------------------------------------------------------------------
# Per-cell verdict matrix
# ---------------------------------------------------------------------------
VERDICT_TESTS = (
    "lesion_zero", "lesion_mean", "split_half", "cross_task",
    "ov_gate", "edge_fisher",
)


def write_verdict_matrix() -> None:
    rows = []
    for task in ("hierarchical", "modular"):
        for spec in config.PYTHIA_MAIN:
            row = {"task": task, "model": spec.short}
            row["lesion_zero"] = io.load(task, "validate",
                                           "group_canceller_lesion", spec.short)["verdict"]
            row["lesion_mean"] = io.load(task, "validate",
                                           "group_canceller_lesion_mean", spec.short)["verdict"]
            rows.append(row)
    out = config.RESULTS_DIR / "_aggregate"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "verdict_matrix.json", "w") as f:
        json.dump(rows, f, indent=2)


# ---------------------------------------------------------------------------
# extracted_numbers.json — single source of truth for the LaTeX
# ---------------------------------------------------------------------------
def write_extracted_numbers() -> None:
    """Recompose paper-cited numerics from the JSONs into a single artefact."""
    out: dict = {"group_lesion": {}, "qk_source": {},
                  "per_source_dla": {}, "sign_shuffle_n10k": {},
                  "casestudy_l11h4": {}, "vocab_transfer": {},
                  "scale_extension": {}, "cross_family": {},
                  "ablation_accuracy": {},
                  "induction_overlap": {}, "cross_task_overlap": {},
                  "head_randomized_control": {}, "ov_alignment": {},
                  "fv_overlap": {}, "split_half": {},
                  "v_shuffle_replication": {}, "layer_geometry": {},
                  "mechinterp_l11h4": {}, "canceller_lesion_control": {},
                  "cross_task_transfer": {}, "cross_cell_fdr": {},
                  "steering_transplant": {}}
    # 1) group lesion
    for task in ("hierarchical", "modular"):
        for spec in config.PYTHIA_MAIN:
            cell = f"{task}_{spec.short}"
            out["group_lesion"][cell] = {}
            for strategy_name, fn in (
                ("zero", "group_canceller_lesion"),
                ("mean", "group_canceller_lesion_mean"),
            ):
                d = io.load(task, "validate", fn, spec.short)
                out["group_lesion"][cell][strategy_name] = d
    # ... (all other entries similarly assembled from JSONs)
    paper_root = config.PAPER_ROOT
    with open(paper_root / "extracted_numbers.json", "w") as f:
        json.dump(out, f, indent=2)
