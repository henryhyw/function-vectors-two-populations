"""Figure 3 — Layer distribution of writers and cancellers.

2x3 grid (hier/mod x {410M,1B,1.4B}). Within each cell, writers (red)
sit on the top lane and cancellers (blue) on the bottom lane, placed
by layer index. Marker area is proportional to the z-scored |DLA|.
Grey lines connect each canceller to its earliest (lowest-layer)
upstream writer, illustrating the 23/27 downstream-of-a-writer claim.

Data: extracted_numbers.json -> layer_geometry.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "serif", "font.size": 7.0,
    "axes.titlesize": 7.2, "axes.labelsize": 7.0,
    "xtick.labelsize": 6.2, "ytick.labelsize": 6.2,
    "figure.dpi": 200,
    "savefig.bbox": "standard", "savefig.pad_inches": 0.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.0, "patch.linewidth": 0.0,
    "mathtext.fontset": "cm",
})

COL_W = "#b2182b"   # writer (red)
COL_C = "#2166ac"   # canceller (blue)
COL_LINK = "#9a9a9a"

ROOT    = Path("/sessions/affectionate-quirky-goodall/mnt/outputs/paper")
NUMBERS = ROOT / "extracted_numbers.json"
OUT     = ROOT / "draft/figures/fig3_layer_geometry.pdf"

CELL_ORDER = [
    ("hier", "hierarchical_410m", "410M"),
    ("hier", "hierarchical_1b",   "1B"),
    ("hier", "hierarchical_1.4b", "1.4B"),
    ("mod",  "modular_410m",      "410M"),
    ("mod",  "modular_1b",        "1B"),
    ("mod",  "modular_1.4b",      "1.4B"),
]

d = json.load(open(NUMBERS))["layer_geometry"]

# Global marker-size scaling from |z| across all heads.
all_absz = []
for _, key, _ in CELL_ORDER:
    for h in d[key]["heads"]:
        all_absz.append(abs(h.get("z", 0.0)))
zmax = max(all_absz) if all_absz else 1.0


def msize(z):
    """Marker area proportional to z-scored |DLA|, with a visible floor."""
    return 14 + 95 * (abs(z) / zmax)


fig, axes = plt.subplots(2, 3, figsize=(7.1, 2.95),
                         sharex=False, constrained_layout=False)
fig.subplots_adjust(left=0.045, right=0.995, top=0.86, bottom=0.14,
                    wspace=0.16, hspace=0.55)

Y_W, Y_C = 1.0, 0.0   # writer lane / canceller lane
DODGE = 0.24          # vertical spread for heads sharing a (layer, lane)


def with_dodge(heads, y0):
    """Assign each head a y-coordinate; co-layer heads in the same lane are
    spread symmetrically around the lane centre so markers never coincide."""
    by_layer = {}
    for h in heads:
        by_layer.setdefault(h["L"], []).append(h)
    pos = {}
    for L, group in by_layer.items():
        n = len(group)
        # symmetric offsets: 0 for singletons, spread for collisions
        offs = [0.0] if n == 1 else np.linspace(-DODGE, DODGE, n)
        for h, o in zip(sorted(group, key=lambda x: x["z"]), offs):
            pos[(h["L"], h["H"])] = y0 + o
    return pos


for ax, (task, key, size) in zip(axes.flat, CELL_ORDER):
    cell = d[key]
    nL = cell["n_layers"]
    W = [h for h in cell["heads"] if h["role"] == "W"]
    C = [h for h in cell["heads"] if h["role"] == "C"]
    yW = with_dodge(W, Y_W)
    yC = with_dodge(C, Y_C)

    # markers
    ax.scatter([w["L"] for w in W], [yW[(w["L"], w["H"])] for w in W],
               s=[msize(w["z"]) for w in W],
               c=COL_W, edgecolors="white", linewidths=0.5, zorder=3)
    ax.scatter([c["L"] for c in C], [yC[(c["L"], c["H"])] for c in C],
               s=[msize(c["z"]) for c in C],
               c=COL_C, edgecolors="white", linewidths=0.5, zorder=3)

    tname = "hier" if task == "hier" else "mod"
    ax.set_title(rf"{tname}$\cdot${size}  $|\mathcal{{W}}|${'='}{len(W)}, "
                 rf"$|\mathcal{{C}}|${'='}{len(C)}, $L${'='}{nL}",
                 fontsize=7.0, pad=3)
    ax.set_xlim(-0.5, nL - 0.5)
    ax.set_ylim(-0.7, 1.7)
    ax.set_yticks([Y_C, Y_W])
    ax.set_yticklabels(["C", "W"], fontsize=6.5)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=2)
    ax.set_xticks([0, nL // 2, nL - 1])
    ax.spines["left"].set_visible(False)

# shared x-label
fig.text(0.52, 0.015, "layer index", ha="center", fontsize=7.0)

# legend
handles = [
    Line2D([0], [0], marker="o", color="none", markerfacecolor=COL_W,
           markeredgecolor="white", markersize=6, label="writer"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor=COL_C,
           markeredgecolor="white", markersize=6, label="canceller"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor="#999999",
           markeredgecolor="white", markersize=4,
           label="marker area $\\propto$ $z$-scored $|$DLA$|$"),
]
fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
           fontsize=6.6, handletextpad=0.4, columnspacing=1.6,
           bbox_to_anchor=(0.52, 1.005))

fig.savefig(OUT)
print(f"wrote {OUT}")
