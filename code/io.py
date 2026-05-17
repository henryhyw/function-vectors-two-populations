"""Per-cell JSON load/save under a stable directory layout.

Layout:

    results/{task_id}/F0/discover/        refined_dla, screened_heads
    results/{task_id}/F0/validate/        path_patching, group_lesion, ...
    results/{task_id}/F0/mechanism/       qk_source, ov_alignment
    results/{task_id}/F0/controls/        induction_top10
    results/{task_id}/_aggregate/         cross-cell summaries
    results/_aux/<theme>/                 auxiliary experiments

Filenames always end with the cell short name, e.g. ``..._410m.json``.
"""
from __future__ import annotations
import json
from pathlib import Path
from . import config


def cell_dir(task_id: str, stage: str) -> Path:
    return config.RESULTS_DIR / task_id / config.FORMAT_ID / stage


def load(task_id: str, stage: str, name: str, model_short: str) -> dict:
    path = cell_dir(task_id, stage) / f"{name}_{model_short}.json"
    with open(path) as f:
        return json.load(f)


def save(task_id: str, stage: str, name: str, model_short: str, payload: dict) -> Path:
    out = cell_dir(task_id, stage)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}_{model_short}.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def aux_save(theme: str, name: str, payload: dict) -> Path:
    """Save an auxiliary-experiment payload under ``results/_aux/<theme>/<name>.json``."""
    out = config.AUX_DIR / theme
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def aux_load(theme: str, name: str) -> dict:
    with open(config.AUX_DIR / theme / f"{name}.json") as f:
        return json.load(f)
