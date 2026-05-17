"""Scale + cross-architecture extension.

Runs the full discovery → validate → group-lesion → steering chain on
six additional cells:

  - Pythia-2.8B / 6.9B / 12B  on hierarchical and modular,
  - Qwen2.5-1.5B / Qwen2.5-7B on modular only (single-task spot check),
  - GPT-2-medium              on modular only.

Returns the four-condition canonical verdict per cell plus the steering
contrast :math:`\\Delta\\ell(v_\\mathcal{W}) - \\Delta\\ell(v_\\text{FV})`.
"""
from __future__ import annotations
from typing import Sequence
from . import dla, path_patching, group_lesion, steering, prompts, io
from .config import N_DISCOVERY, N_PP, N_EVAL, DISCOVERY_SEEDS, EVAL_SEED


def run_one_cell(load_model, model_short: str, task_id: str) -> dict:
    """End-to-end pipeline for one (model, task) cell."""
    model = load_model()
    discov = prompts.generate_batch(task_id, N_DISCOVERY, base_seed=DISCOVERY_SEEDS[0])
    pp_pairs = [prompts.make_paired_rule_flip(task_id, seed=1000 + i) for i in range(N_PP)]
    eval_p = prompts.generate_batch(task_id, N_EVAL, base_seed=EVAL_SEED)

    # 1) refined-DLA discovery
    raw_dla = dla.per_head_refined_dla(model, discov)
    fv = dla.screen_fv_candidates(raw_dla)

    # 2) path patching → W/C/weak partition
    pp = path_patching.path_patch_per_head(model, pp_pairs, fv["fv_heads"])
    partition = path_patching.partition_into_W_C_weak(pp)

    # 3) group lesion + four-condition verdict
    lesion = group_lesion.run(model, eval_p, partition, strategy="zero")

    # 4) build steering vectors (downstream evaluation in revision notebook)
    sv = steering.build_steering_vectors(model, discov, partition)
    return {
        "model": model_short, "task": task_id,
        "fv": fv, "partition": partition,
        "lesion": lesion,
        "steering_vectors": {k: v for k, v in sv.items() if k != "head_means"},
    }
