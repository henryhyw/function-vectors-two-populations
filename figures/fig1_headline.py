"""Figure 1 — Forest plot of group-lesion logit shifts (structural finding)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif", "font.size": 8.0,
    "axes.titlesize": 8.5, "axes.labelsize": 7.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 7.2,
    "figure.dpi": 200,
    "savefig.bbox": "standard", "savefig.pad_inches": 0.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.0, "patch.linewidth": 0.0,
    "mathtext.fontset": "cm", "mathtext.rm": "serif",
})

COL_W = "#b2182b"; COL_C = "#2166ac"; COL_J = "#7f8c8d"

REPO = Path("/sessions/affectionate-quirky-goodall/mnt/outputs")
NUMBERS = REPO / "paper" / "extracted_numbers.json"
OUT = REPO / "paper" / "draft" / "figures" / "fig1_headline.pdf"

with open(NUMBERS) as f:
    nums = json.load(f)

HIER_CELLS = [
    ("hier-410M",  ("group_lesion", "hierarchical_410m")),
    ("hier-1B",    ("group_lesion", "hierarchical_1b")),
    ("hier-1.4B",  ("group_lesion", "hierarchical_1.4b")),
    ("hier-2.8B",  ("scale_extension", "pythia-2.8b_hierarchical")),
    ("hier-6.9B",  ("scale_extension", "pythia-6.9b_hierarchical")),
    ("hier-12B",   ("scale_extension", "pythia-12b_hierarchical")),
]
MOD_CELLS = [
    ("mod-410M",  ("group_lesion", "modular_410m")),
    ("mod-1B",    ("group_lesion", "modular_1b")),
    ("mod-1.4B",  ("group_lesion", "modular_1.4b")),
    ("mod-2.8B",  ("scale_extension", "pythia-2.8b_modular")),
    ("mod-6.9B",  ("scale_extension", "pythia-6.9b_modular")),
    ("mod-12B",   ("scale_extension", "pythia-12b_modular")),
    ("Qwen-1.5B", ("cross_family", "Qwen/Qwen2.5-1.5B")),
    ("Qwen-7B",   ("cross_family", "Qwen/Qwen2.5-7B")),
    ("GPT-2-M",   ("cross_family", "gpt2-medium")),
]


def lookup_lesion(source, key):
    if source == "group_lesion":
        m = nums["group_lesion"][key]["mean"]
        return {"W_shift": m["W_shift"], "W_ci": m["W_ci"],
                "C_shift": m["C_shift"], "C_ci": m["C_ci"],
                "both_shift": m["both_shift"], "both_ci": m["both_ci"]}
    elif source in ("scale_extension", "cross_family"):
        e = nums[source][key]
        return {"W_shift": e["writer_effect"]["mean_logit_shift"],
                "W_ci":    e["writer_effect"].get("ci_95"),
                "C_shift": e["canceller_effect"]["mean_logit_shift"],
                "C_ci":    e["canceller_effect"].get("ci_95"),
                "both_shift": e["joint_effect"]["mean_logit_shift"],
                "both_ci":   e["joint_effect"].get("ci_95")}
    raise ValueError(source)


def render_forest(ax, rows, xlim):
    n = len(rows)
    ys = np.arange(n)[::-1]; gap = 0.22
    for i, r in enumerate(rows):
        y = ys[i]; L = r["lesion"]
        for off, (vk, ck), col in [
            (+gap, ("W_shift","W_ci"), COL_W),
            (0.0,  ("C_shift","C_ci"), COL_C),
            (-gap, ("both_shift","both_ci"), COL_J),
        ]:
            v = L[vk]
            ci = L[ck] if isinstance(L[ck], (list, tuple)) else [v, v]
            ax.plot([ci[0], ci[1]], [y + off, y + off], color=col, linewidth=1.2, alpha=0.85)
            ax.scatter([v], [y + off], color=col, s=15, zorder=3)
    ax.axvline(0, color="black", linewidth=0.5, alpha=0.5)
    ax.set_yticks(ys)
    ax.set_yticklabels([r["label"] for r in rows])
    ax.set_xlim(*xlim)
    ax.set_xlabel(r"$\Delta\ell$ (nats)", labelpad=2)


hier_rows = [{"label": lab, "lesion": lookup_lesion(*args)} for lab, args in HIER_CELLS]
mod_rows  = [{"label": lab, "lesion": lookup_lesion(*args)} for lab, args in MOD_CELLS]


def lim(rs):
    vs = []
    for r in rs:
        for k, ck in [("W_shift","W_ci"), ("C_shift","C_ci"), ("both_shift","both_ci")]:
            ci = r["lesion"][ck]
            if isinstance(ci, (list, tuple)): vs += list(ci)
            vs.append(r["lesion"][k])
    lo, hi = min(vs), max(vs); pad = 0.05 * (hi - lo)
    return (lo - pad, hi + pad)


row_h = 0.20
fig_h = max(len(hier_rows), len(mod_rows)) * row_h + 0.75
fig = plt.figure(figsize=(7.0, fig_h), constrained_layout=True)
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.08)

axHL = fig.add_subplot(gs[0])
axML = fig.add_subplot(gs[1])

render_forest(axHL, hier_rows, lim(hier_rows))
render_forest(axML, mod_rows,  lim(mod_rows))

axHL.set_title(r"$\bf{(a)}$ hierarchical", fontsize=8.5, loc="left", pad=4, x=0.0)
axML.set_title(r"$\bf{(b)}$ modular + cross-architecture", fontsize=8.5, loc="left", pad=4, x=0.0)

legend = [
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COL_W,
                markersize=5, label=r"ablate $\mathcal{W}$"),
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COL_C,
                markersize=5, label=r"ablate $\mathcal{C}$"),
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COL_J,
                markersize=5, label=r"joint $\mathcal{W}\!\cup\!\mathcal{C}$"),
]
fig.legend(handles=legend, loc="outside lower center", ncol=3, frameon=False,
            fontsize=7.5, handletextpad=0.4, columnspacing=1.6)

fig.savefig(OUT)
print(f"wrote {OUT}")
