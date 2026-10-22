"""Manuscript figures, generated only from results/ run files.

Colour roles (validated categorical slots 1-3 + neutral):
  tabular = blue #2a78d6, graph model = orange #eb6834, hybrid = aqua #1baf7a,
  reference = gray. Differences use a blue<->red diverging map with a gray midpoint.
Every series also has its own marker and a direct label, so identity is never colour-only.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from ..ledger import RESULTS_DIR
from ..metrics import t_ci
from .analyze import load_rows

FIG = RESULTS_DIR / "figures"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
DIVERGING = LinearSegmentedColormap.from_list("bgr", ["#184f95", "#6da7ec", "#f0efec", "#ec8a89", "#a32a2a"])
STYLE = {"GB-CA": (BLUE, "o", "-"), "GB-CAG": (BLUE, "s", "--"), "HGT": (ORANGE, "^", "-"),
         "SAGE": (ORANGE, "D", "--"), "HGT+AG": (AQUA, "v", "-"), "Overlap-score": (GRAY, "x", ":"),
         "MLP-C": (GRAY, "P", "-")}

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "lines.linewidth": 1.6, "savefig.dpi": 300,
    "pdf.fonttype": 42, "font.family": "DejaVu Sans",
})


def _ci_half(x):
    lo, hi = t_ci(np.asarray(x))
    return (hi - lo) / 2 if np.isfinite(lo) else 0.0


def regime_lines(df: pd.DataFrame) -> None:
    methods = ["GB-CA", "GB-CAG", "HGT", "SAGE", "HGT+AG"]
    shifts = sorted(df.ring_shift.unique())
    fig, axes = plt.subplots(1, len(shifts), figsize=(7.0, 2.3), sharey=True)
    for ax, shift in zip(np.atleast_1d(axes), shifts):
        cell = df[df.ring_shift == shift]
        for m in methods:
            g = cell[cell.method == m].groupby("ring_density")["ring_ap"]
            mean, half = g.mean(), g.apply(_ci_half)
            color, marker, ls = STYLE[m]
            ax.errorbar(mean.index, mean.values, yerr=half.values, color=color, marker=marker, ls=ls,
                        ms=4.5, capsize=2, elinewidth=0.8, label=m, mec="white", mew=0.6)
        ax.set_title(f"feature shift $\\delta$ = {shift:g}", color=INK)
        ax.set_xlabel("structural density $s$")
        ax.set_xticks(sorted(df.ring_density.unique()))
    lo = df[df.method.isin(methods)].groupby(["ring_shift", "ring_density", "method"])["ring_ap"].mean().min()
    np.atleast_1d(axes)[0].set_ylim(max(0.0, np.floor((lo - 0.03) * 20) / 20), 1.0)
    np.atleast_1d(axes)[0].set_ylabel("ring AP (unseen rings)")
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(methods), frameon=False, bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIG / "fig_regime_ring_ap.pdf", bbox_inches="tight")
    plt.close(fig)


def delta_heatmaps(df: pd.DataFrame, comp: pd.DataFrame) -> None:
    pairs = [("HGT", "GB-CAG"), ("SAGE", "GB-CAG"), ("HGT", "GB-CA")]
    shifts, dens = sorted(df.ring_shift.unique()), sorted(df.ring_density.unique())
    fig, axes = plt.subplots(1, len(pairs), figsize=(7.0, 2.4))
    lim = 0.0
    mats = []
    for m, ref in pairs:
        c = comp[(comp.method == m) & (comp.reference == ref)]
        M = np.full((len(shifts), len(dens)), np.nan)
        P = np.full_like(M, np.nan)
        for _, r in c.iterrows():
            M[shifts.index(r.ring_shift), dens.index(r.ring_density)] = r.mean_diff
            P[shifts.index(r.ring_shift), dens.index(r.ring_density)] = r.t_p_holm
        mats.append((M, P))
        lim = max(lim, np.nanmax(np.abs(M)))
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-lim, vmax=lim)
    for ax, (m, ref), (M, P) in zip(axes, pairs, mats):
        im = ax.imshow(M, cmap=DIVERGING, norm=norm, aspect="auto", origin="lower")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isfinite(M[i, j]):
                    star = "*" if P[i, j] < 0.05 else ""
                    ax.text(j, i, f"{M[i, j]:+.2f}{star}", ha="center", va="center", fontsize=6.5,
                            color="white" if abs(M[i, j]) > 0.6 * lim else INK)
        ax.set_xticks(range(len(dens)), [f"{d:g}" for d in dens])
        ax.set_yticks(range(len(shifts)), [f"{s:g}" for s in shifts])
        ax.set_xlabel("structural density $s$")
        ax.set_title(f"{m} $-$ {ref}", color=INK)
        ax.grid(False)
    axes[0].set_ylabel("feature shift $\\delta$")
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("$\\Delta$ ring AP", color=INK)
    fig.savefig(FIG / "fig_delta_heatmaps.pdf", bbox_inches="tight")
    plt.close(fig)


def ablation_dots(abl: pd.DataFrame) -> None:
    order = ["full", "rewired", "no_visits", "no_degree", "layers_1", "layers_2"]
    tick = {"full": "full (3 layers)", "rewired": "rewired", "no_visits": "no visits relation",
            "no_degree": "no degree enc.", "layers_1": "1 layer", "layers_2": "2 layers"}
    dens_values = sorted(abl.ring_density.unique())
    fig, axes = plt.subplots(1, len(dens_values), figsize=(7.0, 2.6), sharey=True)
    methods = [("GB-CAG", BLUE, "s"), ("SAGE", ORANGE, "D"), ("HGT", ORANGE, "^")]
    for ax, d in zip(np.atleast_1d(axes), dens_values):
        sub = abl[abl.ring_density == d]
        for k, (m, color, marker) in enumerate(methods):
            g = sub[sub.method == m]
            vs = [v for v in order if v in set(g.variant)]
            for v in vs:
                x = order.index(v) + (k - 1) * 0.22
                vals = g[g.variant == v]["ring_ap"]
                ax.errorbar(x, vals.mean(), yerr=_ci_half(vals), color=color, marker=marker, ms=4.5,
                            capsize=2, elinewidth=0.8, mec="white", mew=0.6, ls="none",
                            label=m if v == "full" else None)
        ax.set_xticks(range(len(order)), [tick[o] for o in order], rotation=30, ha="right")
        ax.set_title(f"$\\delta$ = 0, $s$ = {d:g}", color=INK)
        ax.set_ylim(0.4, 1.01)
    np.atleast_1d(axes)[0].set_ylabel("ring AP (unseen rings)")
    np.atleast_1d(axes)[0].legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "fig_ablation.pdf", bbox_inches="tight")
    plt.close(fig)


def kaggle_dots(kag: pd.DataFrame) -> None:
    g = kag.groupby("method")["ap"]
    stats = pd.DataFrame({"mean": g.mean(), "half": g.apply(_ci_half)}).sort_values("mean")
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    for i, (m, r) in enumerate(stats.iterrows()):
        color = AQUA if "+" in m else (BLUE if m.split("-")[0] in ("LR", "RF", "GB") else (GRAY if m.startswith("MLP") else ORANGE))
        ax.errorbar(r["mean"], i, xerr=r["half"], color=color, marker="o", ms=4, capsize=2, elinewidth=0.8, ls="none")
        ax.text(r["mean"] + r["half"] + 0.005, i, f"{r['mean']:.3f}", va="center", fontsize=6.5, color=INK2)
    ax.set_yticks(range(len(stats)), stats.index)
    ax.set_xlabel("provider-level AP (mean, 95% CI over 5 folds)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_kaggle_ap.pdf", bbox_inches="tight")
    plt.close(fig)


def protocol_pairs(t: pd.DataFrame) -> None:
    methods = ["MLP-C", "GB-CA", "GB-CAG", "SAGE", "HGT", "Overlap-score"]
    cells = t[["ring_shift", "ring_density"]].drop_duplicates().sort_values(["ring_shift", "ring_density"])
    fig, axes = plt.subplots(1, len(cells), figsize=(7.0, 2.3), sharey=True)
    for ax, (_, c) in zip(np.atleast_1d(axes), cells.iterrows()):
        sub = t[(t.ring_shift == c.ring_shift) & (t.ring_density == c.ring_density)]
        for i, m in enumerate(methods):
            e = sub[(sub.method == m) & (sub.protocol == "emergent")]["mean"]
            p = sub[(sub.method == m) & (sub.protocol == "persistent")]["mean"]
            if e.empty or p.empty:
                continue
            color = STYLE[m][0]
            ax.plot([i, i], [e.iloc[0], p.iloc[0]], color=GRID, lw=2, zorder=1)
            ax.scatter([i], [e.iloc[0]], color=color, marker="o", s=24, zorder=2, label="new rings" if i == 0 else None)
            ax.scatter([i], [p.iloc[0]], edgecolors=color, marker="s", s=24, zorder=2, facecolors="white",
                       linewidths=1.2, label="recurring rings" if i == 0 else None)
        ax.set_xticks(range(len(methods)), [m.replace("Overlap-score", "Overlap") for m in methods], rotation=30, ha="right")
        ax.set_title(f"$\\delta$ = {c.ring_shift:g}, $s$ = {c.ring_density:g}", color=INK)
        ax.set_ylim(0, 1.02)
    np.atleast_1d(axes)[0].set_ylabel("ring AP")
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(FIG / "fig_protocol.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    syn = load_rows("synthetic")
    if not syn.empty:
        regime_lines(syn)
        comp_path = RESULTS_DIR / "tables" / "synthetic_ring_ap_comparisons.csv"
        if comp_path.exists():
            delta_heatmaps(syn, pd.read_csv(comp_path))
    abl = load_rows("ablation")
    if not abl.empty:
        ablation_dots(abl)
    kag = load_rows("kaggle")
    if not kag.empty:
        kaggle_dots(kag)
    prot = RESULTS_DIR / "tables" / "protocol_ring_ap.csv"
    if prot.exists():
        protocol_pairs(pd.read_csv(prot))
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
