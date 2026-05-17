"""Writers and Cancellers reproducibility package.

The full experimental pipeline is driven by ``pipeline.ipynb`` at the
repository root.  This package exposes the building blocks that the
notebook composes:

  - :mod:`config`       — model list, task definitions, fixed thresholds.
  - :mod:`prompts`      — deterministic prompt generators (4-shot ICL).
  - :mod:`utils`        — bootstrap, FDR, BCa CI, deterministic seeding.
  - :mod:`dla`          — refined direct logit attribution.
  - :mod:`path_patching`— per-head path-patching causal effects.
  - :mod:`group_lesion` — W/C/joint ablation + four-condition verdict.
  - :mod:`qk_source`    — QK source-bucket decomposition.
  - :mod:`per_source_dla` — content-vs-sink DLA attribution.
  - :mod:`rule_outs`    — rank-1, V-cascade, V-shuffle, induction, head-rand.
  - :mod:`steering`     — :math:`v_\\text{FV}` vs :math:`v_\\mathcal{W}` vs PCA.
  - :mod:`cross_template` — vocabulary-ICL transfer (antonym, country-capital).
  - :mod:`scale_extension` — Pythia 2.8B/6.9B/12B + Qwen + GPT-2-medium.
  - :mod:`case_study`   — single-head L11.H4 V-shuffle / OV / V-composition.
  - :mod:`io`           — load/save JSON results under :data:`config.RESULTS_DIR`.
"""
from . import config, prompts, utils, io  # noqa: F401
