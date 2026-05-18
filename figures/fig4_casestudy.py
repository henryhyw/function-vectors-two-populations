"""Figure 4 — L11.H4 case study: cross-template sign-flip.

Single-panel: L11.H4 solo-ablation Δℓ across four templates.
The head's role flips canceller → writer when transferred from
the rule tasks (hier, mod) to antonym; null on country-capital.

Numerics in App. F (L11.H4 solo ablation on vocabulary tasks).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.patches import Patch

plt.rcParams.update({
    "font.family": "serif", "font.size": 7.5,
    "axes.titlesize": 8.5, "axes.labelsize": 7.5,
    "xtick.labelsize": 6.8, "ytick.labelsize": 7.0,
    "figure.dpi": 200,
    "savefig.bbox": "standard", "savefig.pad_inches": 0.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.0, "patch.linewidth": 0.0,
    "mathtext.fontset": "cm",
})

COL_C    = "#2166ac"   # canceller (blue)
COL_W    = "#b2182b"   # writer (red)
COL_NULL = "#999999"

OUT = Path("/sessions/affectionate-quirky-goodall/mnt/outputs/paper/draft/figures/fig4_casestudy.pdf")

# Compact single-panel figure.
fig, ax = plt.subplots(figsize=(3.4, 1.40), constrained_layout=False)
fig.subplots_adjust(left=0.15, right=0.985, top=0.93, bottom=0.30)

# Cross-template sign-flip: L11.H4 solo ablation Δℓ per template.
templates = ["Hier", "Mod", "Antonym", "Cnt-Cap"]
shifts = [+0.216, +0.215, -0.069, -0.005]
ci_lo  = [+0.190, +0.189, -0.109, -0.033]
ci_hi  = [+0.242, +0.241, -0.028, +0.025]
roles  = ["canceller", "canceller", "writer", "null"]
colors = [COL_C if r == "canceller"
           else (COL_W if r == "writer" else COL_NULL)
           for r in roles]
xs = np.arange(len(templates))
ax.bar(xs, shifts, color=colors, width=0.62)
ax.errorbar(xs, shifts,
             yerr=[np.array(shifts) - np.array(ci_lo),
                   np.array(ci_hi) - np.array(shifts)],
             fmt="none", ecolor="black", elinewidth=0.7, capsize=2.0)
ax.axhline(0, color="black", lw=0.5, alpha=0.6)
ax.set_xticks(xs)
ax.set_xticklabels(templates, rotation=20, ha="right", fontsize=6.8)
ax.set_ylabel(r"L11.H4 solo $\Delta\ell$ (nats)", labelpad=1, fontsize=7)
ax.set_ylim(-0.18, 0.32)
ax.tick_params(axis="y", labelsize=6.5)
ax.legend(handles=[Patch(color=COL_C,   label="canceller"),
                    Patch(color=COL_W,   label="writer"),
                    Patch(color=COL_NULL, label="null")],
           loc="upper right", frameon=False, fontsize=6,
           handletextpad=0.3, labelspacing=0.18,
           borderpad=0.1, borderaxespad=0.3)

fig.savefig(OUT)
print(f"wrote {OUT}")
