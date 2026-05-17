"""Pre-registered configuration: models, tasks, thresholds, paths.

All numbers fixed in advance of the primary analysis. Changes here
require a new format version (we use ``F0`` throughout).
"""
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PAPER_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PAPER_ROOT / "results"
AUX_DIR = RESULTS_DIR / "_aux"
FORMAT_ID = "F0"  # only one frozen format used throughout


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelSpec:
    name: str
    short: str
    arch: str          # "neox" | "llama" | "conv1d"
    n_layers: int
    d_model: int
    n_heads: int
    dtype: str = "fp32"


PYTHIA_MAIN = [
    ModelSpec("EleutherAI/pythia-410m", "410m",  "neox",  24, 1024, 16),
    ModelSpec("EleutherAI/pythia-1b",   "1b",    "neox",  16, 2048,  8),
    ModelSpec("EleutherAI/pythia-1.4b", "1.4b",  "neox",  24, 2048, 16),
]

PYTHIA_LADDER_EXT = [
    ModelSpec("EleutherAI/pythia-2.8b", "2.8b",  "neox",  32, 2560, 32, dtype="fp32"),
    ModelSpec("EleutherAI/pythia-6.9b", "6.9b",  "neox",  32, 4096, 32, dtype="bf16"),
    ModelSpec("EleutherAI/pythia-12b",  "12b",   "neox",  36, 5120, 40, dtype="bf16"),
]

CROSS_FAMILY = [
    ModelSpec("Qwen/Qwen2.5-1.5B", "qwen-1.5b", "llama",   28, 1536, 12),
    ModelSpec("Qwen/Qwen2.5-7B",   "qwen-7b",   "llama",   28, 3584, 28, dtype="bf16"),
    ModelSpec("gpt2-medium",       "gpt2-m",    "conv1d",  24, 1024, 16),
]


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TaskSpec:
    task_id: str           # "hierarchical" | "modular" | "antonym" | "country-capital"
    n_features: int = 8
    n_demos: int = 4
    label_repetition_beta: float = 1.0  # 1.0 = no repetition allowed across demos


HIER = TaskSpec("hierarchical")
MOD = TaskSpec("modular")
VOCAB_TASKS = ["antonym", "country-capital"]
RULE_TASKS = [HIER, MOD]


# ---------------------------------------------------------------------------
# Sample sizes / seeds
# ---------------------------------------------------------------------------
N_DISCOVERY = 192     # discovery prompts for refined-DLA + DLA permutations
N_PP = 200            # paired path-patching prompts
N_EVAL = 500          # paired eval prompts (group lesion + steering)
N_VOCAB_EVAL = 100    # paired eval for cross-template transfer
N_CASESTUDY = 200     # paired prompts for L11.H4 V-shuffle / V-composition

DISCOVERY_SEEDS = (42, 43)
PP_SEED = 43
EVAL_SEED = 44
PERM_SEED = 12345


# ---------------------------------------------------------------------------
# Thresholds (frozen)
# ---------------------------------------------------------------------------
DLA_FDR_Q = 0.10              # BH-FDR cutoff for refined-DLA discovery
PP_DIRECT_GATE = 0.05         # |direct effect| ≥ 5% to enter W ∪ C
LESION_MIN_SHIFT = 0.10       # canonical-verdict floor (i)/(ii) on |Δℓ|
SHUFFLE_NPERM = 10_000        # FV-set sign-shuffle null
DLA_NPERM = 20                # refined-DLA label permutations
BOOTSTRAP_B = 10_000          # paired-prompt bootstrap iterations
JACKKNIFE_OUTLOO = True       # leave-one-out for geometry CIs
RULE_SPECIFIC_GATE = 5.0      # rule-NLL/random-NLL ratio threshold
GENERIC_GATE = 1.5            # ratio threshold for "generally important"


# ---------------------------------------------------------------------------
# Steering grid
# ---------------------------------------------------------------------------
STEER_ALPHA_GRID = (0.5, 1.0, 2.0, 4.0, 8.0)
STEER_LAYER_FRAC = 0.5  # inject at residual stream after layer L*= n_layers // 2
