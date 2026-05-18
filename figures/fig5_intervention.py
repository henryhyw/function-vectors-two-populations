"""Figure 5: Canceller ablation — actionable consequence.

Per-cell forest plot of accuracy delta after zero-ablating the
canceller subgroup, with paired bootstrap 95% CI. Annotated with
the logit-shift magnitude per cell (App. D.3 Tab. tab:accuracy).
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif", "font.size": 7.5,
    "axes.titlesize": 8.0, "axes.labelsize": 7.5,
    "xtick.labelsize": 7.0, "ytick.labelsize": 7.5,
    "figure.dpi": 200,
    "savefig.bbox": "standard", "savefig.pad_inches": 0.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.0, "patch.linewidth": 0.0,
    "mathtext.fontset": "cm",
})

COL_C = "#2166ac"   # canceller (matches fig 2)
COL_TEXT = "#2a2a28"

ROOT = Path("/sessions/affectionate-quirky-goodall/mnt/outputs/paper")
NUMBERS = ROOT / "extracted_numbers.json"
OUT     = ROOT / "draft/figures/fig5_intervention.pdf"


CELL_ORDER = [
    ("hier-410M",  "hierarchical_410m"),
    ("hier-1B",    "hierarchical_1b"),
    ("hier-1.4B",  "hierarchical_1.4b"),
    ("mod-410M",   "modular_410m"),
    ("mod-1B",     "modular_1b"),
    ("mod-1.4B",   "modular_1.4b"),
]

d = json.load(open(NUMBERS))
rows = []
for label, key in CELL_ORDER:
    a = d["ablation_accuracy"][key]
    C = a["C"]
    rows.append({
        "label":    label,
        "baseline": a["baseline_acc"],
        "delta":    C["delta_acc"],
        "ci":       C["ci_95"],
        "logit":    C["logit_shift"],
    })

fig, ax = plt.subplots(figsize=(3.4, 1.55), constrained_layout=True)

n = len(rows)
y_pos = np.arange(n)[::-1]   # top row = hier-410M

for i, r in enumerate(rows):
    yp = y_pos[i]
    # CI horizontal line
    ax.plot([r["ci"][0], r["ci"][1]], [yp, yp],
             color=COL_C, linewidth=2.0, alpha=0.55, solid_capstyle="round")
    # bookend ticks
    for v in r["ci"]:
        ax.plot([v, v], [yp - 0.18, yp + 0.18],
                 color=COL_C, linewidth=1.2, alpha=0.85)
    # point
    ax.plot(r["delta"], yp, "o",
             markersize=5, markerfacecolor=COL_C,
             markeredgecolor="white", markeredgewidth=1.0)
    # logit shift annotation on the right
    ax.text(0.165, yp,
             f"  {r['logit']:+.2f} nats",
             va="center", ha="left", fontsize=6.6,
             color=COL_TEXT, family="monospace", alpha=0.85)

# vertical reference line at zero
ax.axvline(0, color="black", linewidth=0.6, alpha=0.55, linestyle=(0, (2, 3)))

ax.set_yticks(y_pos)
ax.set_yticklabels([r["label"] for r in rows], fontsize=7.0)
ax.set_xlim(-0.07, 0.17)
ax.set_xticks([-0.05, 0.00, 0.05, 0.10, 0.15])
ax.set_xticklabels([f"{v*100:+.0f} pp" for v in ax.get_xticks()], fontsize=6.8)
ax.set_xlabel(r"accuracy change after ablating $\mathcal{C}$",
               fontsize=7.0, labelpad=2)

ax.tick_params(axis="x", length=2)
ax.tick_params(axis="y", length=0)

fig.savefig(OUT)
print(f"wrote {OUT}")
