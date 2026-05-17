"""Figure 2: Writers concentrate on demonstration labels;
   cancellers shift mass to format-prefix tokens.

   For each cell, two grouped bars per source bucket show the mean
   per-head attention mass at the readout token: writers (red) and
   cancellers (teal). Directly readable — no subtraction required.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

plt.rcParams.update({
    "font.family": "serif", "font.size": 7.5,
    "axes.titlesize": 8.0, "axes.labelsize": 7.5,
    "xtick.labelsize": 6.8, "ytick.labelsize": 7.0,
    "figure.dpi": 200,
    "savefig.bbox": "standard", "savefig.pad_inches": 0.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.0, "patch.linewidth": 0.0,
    "mathtext.fontset": "cm",
})

COL_W = "#b2182b"   # writer (red)
COL_C = "#2166ac"   # canceller (teal-blue)

QK_DIR = Path("/sessions/affectionate-quirky-goodall/mnt/outputs/paper/results")
OUT    = Path("/sessions/affectionate-quirky-goodall/mnt/outputs/paper/draft/figures/fig2_qk_mechanism.pdf")

CELL_ORDER = [
    ("hier-410M",  "hierarchical/F0/mechanism/qk_source_410m.json"),
    ("hier-1B",    "hierarchical/F0/mechanism/qk_source_1b.json"),
    ("hier-1.4B",  "hierarchical/F0/mechanism/qk_source_1.4b.json"),
    ("mod-410M",   "modular/F0/mechanism/qk_source_410m.json"),
    ("mod-1B",     "modular/F0/mechanism/qk_source_1b.json"),
    ("mod-1.4B",   "modular/F0/mechanism/qk_source_1.4b.json"),
]

BUCKETS       = ["BOS", "format_prefix", "demo_input", "demo_label", "query_input"]
BUCKET_LABELS = ["BOS", "fmt", "d-in", "d-lab", "q-in"]


def load_means(path):
    """Return ((w_mean, c_mean) per bucket as fractions of total per-head mass)."""
    d = json.load(open(QK_DIR / path))
    nW, nC = d["n_writers"], d["n_cancellers"]
    w_raw = [d["writer_bucket_totals"][b]    / nW for b in BUCKETS]
    c_raw = [d["canceller_bucket_totals"][b] / nC for b in BUCKETS]
    # Each head's attention mass sums to 1 over buckets, so the per-head means
    # also sum to ~1. We don't re-normalise (preserve the raw fractions).
    return w_raw, c_raw


cells = [(label, *load_means(path)) for label, path in CELL_ORDER]

n_cells, n_b = len(cells), len(BUCKETS)
fig, axes = plt.subplots(1, n_cells, figsize=(5.5, 1.7),
                          sharey=True, constrained_layout=True)

xs    = np.arange(n_b)
width = 0.36

for ax, (label, w_vals, c_vals) in zip(axes, cells):
    ax.bar(xs - width / 2, w_vals, width=width, color=COL_W, label="W")
    ax.bar(xs + width / 2, c_vals, width=width, color=COL_C, label="C")
    ax.set_xticks(xs)
    ax.set_xticklabels(BUCKET_LABELS, rotation=45, ha="right", fontsize=6.5)
    ax.set_title(label, fontsize=7.5, pad=2)
    ax.tick_params(axis="x", length=0, pad=1)
    ax.set_ylim(0, 0.55)

axes[0].set_ylabel("mean attn mass", fontsize=7.0, labelpad=2)

# Combined legend at bottom
fig.legend(
    handles=[
        Patch(color=COL_W, label=r"writers $(\mathcal{W})$"),
        Patch(color=COL_C, label=r"cancellers $(\mathcal{C})$"),
    ],
    loc="outside lower center", ncol=2,
    frameon=False, fontsize=7.5, handletextpad=0.4, columnspacing=1.4,
)

fig.savefig(OUT)
print(f"wrote {OUT}")
