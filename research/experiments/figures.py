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
from matplotlib.lines import Line2D
from scipy.stats import t as stats_t
from sklearn.metrics import precision_recall_curve

from ..ledger import RESULTS_DIR
from ..metrics import t_ci, val_threshold
from ..synth.generator import GeneratorConfig, generate
from .analyze import DOSE_SETS, ROUND1_SEEDS, load_cascade, load_rows
from .cascade import ESCALATION, route
from .synthetic import run_tag

FIG = RESULTS_DIR / "figures"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
DIVERGING = LinearSegmentedColormap.from_list("bgr", ["#184f95", "#6da7ec", "#f0efec", "#ec8a89", "#a32a2a"])
STYLE = {"GB-CA": (BLUE, "o", "-"), "GB-CAG": (BLUE, "s", "--"), "HGT": (ORANGE, "^", "-"),
         "SAGE": (ORANGE, "D", "--"), "HGT+AG": (AQUA, "v", "-"), "Overlap-score": (GRAY, "x", ":"),
         "MLP-C": (GRAY, "P", "-")}

plt.rcParams.update({
    # HISS recommends Helvetica/Arial lettering of about 8-12 pt; figures are 7 in wide, the journal's full width.
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "lines.linewidth": 1.6, "savefig.dpi": 300,
    "pdf.fonttype": 42, "font.family": ["Arial", "Helvetica", "DejaVu Sans"],
    # Symbols such as delta in labels are set as math text, which otherwise falls back to DejaVu Sans.
    "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
    "mathtext.sf": "Arial",
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
                    label = f"{M[i, j]:+.2f}".replace("+0.00", "0.00").replace("-0.00", "0.00")
                    ax.text(j, i, f"{label}{star}", ha="center", va="center", fontsize=8,
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


def _corrected_half(x, test_train_ratio: float = 1 / 3) -> float:
    """Half-width of the 95% interval over cross-validation rotations with overlapping training sets
    (variance inflated by 1/J + n_test/n_train; Nadeau and Bengio, 2003)."""
    x = np.asarray(x, float)
    return float(stats_t.ppf(0.975, len(x) - 1) * np.sqrt(1 / len(x) + test_train_ratio) * x.std(ddof=1))


def kaggle_dots(kag: pd.DataFrame) -> None:
    g = kag.groupby("method")["ap"]
    stats = pd.DataFrame({"mean": g.mean(), "half": g.apply(_corrected_half)}).sort_values("mean")
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    for i, (m, r) in enumerate(stats.iterrows()):
        color = AQUA if "+" in m else (BLUE if m.split("-")[0] in ("LR", "RF", "GB") else (GRAY if m.startswith("MLP") else ORANGE))
        ax.errorbar(r["mean"], i, xerr=r["half"], color=color, marker="o", ms=4, capsize=2, elinewidth=0.8, ls="none")
        ax.text(r["mean"] + r["half"] + 0.005, i, f"{r['mean']:.3f}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(range(len(stats)), stats.index)
    ax.set_xlabel("provider-level AP (mean, corrected 95% CI)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_kaggle_ap.pdf", bbox_inches="tight")
    plt.close(fig)


def dose_response(df: pd.DataFrame) -> None:
    """Ring AP against ring volume at delta = 0 (seeds 1-6), one panel per structural density."""
    methods = ["GB-CA", "GB-CAG", "SAGE", "HGT", "SAGE+AG", "Overlap-score"]
    style = {**STYLE, "SAGE+AG": (AQUA, "v", "--")}
    dens_values = sorted(df.ring_density.unique())
    fig, axes = plt.subplots(1, len(dens_values), figsize=(7.0, 2.5), sharey=True)
    for ax, d in zip(np.atleast_1d(axes), dens_values):
        sub = df[df.ring_density == d]
        for m in methods:
            g = sub[sub.method == m].groupby("volume")["ring_ap"]
            mean, half = g.mean(), g.apply(_ci_half)
            color, marker, ls = style[m]
            ax.errorbar(mean.index, mean.values, yerr=half.values, color=color, marker=marker, ls=ls, ms=4,
                        capsize=2, elinewidth=0.8, label=m.replace("Overlap-score", "Overlap score"), mec="white", mew=0.6)
        chance = (sub.n_ring / (sub.n_test - sub.n_isolated)).groupby(sub.volume).mean()
        ax.plot(chance.index, chance.values, color=GRAY, lw=0.9, ls=":", label="chance")
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(df.volume.unique()), [f"{v:g}" for v in sorted(df.volume.unique())])
        ax.minorticks_off()
        ax.set_xlabel("ring claims per month")
        ax.set_title(f"$\\delta$ = 0, $s$ = {d:g}", color=INK)
        ax.set_ylim(0, 1.02)
    np.atleast_1d(axes)[0].set_ylabel("ring AP (unseen rings)")
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, -0.08),
               fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIG / "fig_dose_response.pdf", bbox_inches="tight")
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


# Outcome colours for the ring figure (validated all-pairs, light surface): ground truth is carried by
# the nodes (filled = ring member), model outcomes by the edges (colour + line style).
CAUGHT, MISSED, FALSE_ALARM, QUIET = "#6250d6", "#e34948", "#eda100", "#c3c2b7"


def _pair_table(claims: pd.DataFrame, flagged: np.ndarray, ft: np.ndarray) -> pd.DataFrame:
    """One row per patient-doctor pair: claim counts by type and how many of each were flagged."""
    c = claims.assign(ring=(ft == "ring").astype(int), iso=(ft == "isolated").astype(int),
                      legit=(ft == "legit").astype(int), flag=flagged.astype(int))
    c["ring_flag"], c["iso_flag"], c["legit_flag"] = c.ring * c.flag, c.iso * c.flag, c.legit * c.flag
    return c.groupby(["patient_id", "doctor_id"])[["ring", "iso", "legit", "ring_flag", "iso_flag", "legit_flag"]].sum().reset_index()


def _edge_style(row) -> tuple[str, str, float, int]:
    n_fraud, n_caught = row.ring + row.iso, row.ring_flag + row.iso_flag
    width = min(0.8 + 0.35 * (row.ring + row.iso + row.legit), 3.2)
    if n_fraud > 0:
        return (CAUGHT, "-", width, 3) if n_caught >= n_fraud / 2 else (MISSED, (0, (3, 1.5)), width, 3)
    if row.legit_flag > 0:
        return FALSE_ALARM, (0, (1, 1)), max(width, 1.6), 2
    return QUIET, "-", 0.6, 1


def _ring_layout(pairs: pd.DataFrame, ring_docs, ring_pats, rng, max_context: int = 4) -> tuple[dict, pd.DataFrame]:
    pos = {}
    for j, d in enumerate(sorted(ring_docs)):
        a = 2 * np.pi * j / len(ring_docs) + np.pi / 4
        pos[("d", d)] = 0.38 * np.array([np.cos(a), np.sin(a)])
    pat_angle = {}
    for i, p in enumerate(sorted(ring_pats)):
        pat_angle[p] = 2 * np.pi * i / len(ring_pats)
        pos[("p", p)] = 0.82 * np.array([np.cos(pat_angle[p]), np.sin(pat_angle[p])])
    keep = []
    # other doctors of ring patients
    other_docs = pairs[pairs.patient_id.isin(ring_pats) & ~pairs.doctor_id.isin(ring_docs)]
    for d, g in other_docs.groupby("doctor_id"):
        a = np.angle(np.mean(np.exp(1j * np.array([pat_angle[p] for p in g.patient_id]))))
        pos[("d", d)] = 1.25 * np.array([np.cos(a), np.sin(a)])
        keep.append(g)
    # a sample of the ring doctors' other patients
    for j, d in enumerate(sorted(ring_docs)):
        g = pairs[(pairs.doctor_id == d) & ~pairs.patient_id.isin(ring_pats)]
        if len(g) > max_context:
            g = g.iloc[np.sort(rng.choice(len(g), max_context, replace=False))]
        base = 2 * np.pi * j / len(ring_docs) + np.pi / 4
        for k, p in enumerate(g.patient_id):
            a = base + (k - (len(g) - 1) / 2) * 0.16
            pos[("p", p)] = 1.42 * np.array([np.cos(a), np.sin(a)])
        keep.append(g)
    keep.append(pairs[pairs.patient_id.isin(ring_pats) & pairs.doctor_id.isin(ring_docs)])
    return pos, pd.concat(keep).drop_duplicates(["patient_id", "doctor_id"])


def _isolated_layout(pairs: pd.DataFrame, pat: int, doc: int, rng, max_context: int = 10) -> tuple[dict, pd.DataFrame]:
    pos = {("p", pat): np.array([-0.4, 0.0]), ("d", doc): np.array([0.4, 0.0])}
    keep = [pairs[(pairs.patient_id == pat) & (pairs.doctor_id == doc)]]
    g = pairs[(pairs.patient_id == pat) & (pairs.doctor_id != doc)]
    for k, d in enumerate(g.doctor_id):
        a = np.pi + (k - (len(g) - 1) / 2) * 0.5
        pos[("d", d)] = pos[("p", pat)] + 0.8 * np.array([np.cos(a), np.sin(a)])
    keep.append(g)
    g = pairs[(pairs.doctor_id == doc) & (pairs.patient_id != pat)]
    if len(g) > max_context:
        g = g.iloc[np.sort(rng.choice(len(g), max_context, replace=False))]
    for k, p in enumerate(g.patient_id):
        a = (k - (len(g) - 1) / 2) * (2.2 / max(len(g) - 1, 1))
        pos[("p", p)] = pos[("d", doc)] + 0.95 * np.array([np.cos(a), np.sin(a)])
    keep.append(g)
    return pos, pd.concat(keep)


def _draw_panel(ax, pos, edges: pd.DataFrame, members: set, highlight=None) -> None:
    styled = [(r, *_edge_style(r)) for r in edges.itertuples()]
    for r, color, ls, lw, z in sorted(styled, key=lambda x: x[4]):
        a, b = pos[("p", r.patient_id)], pos[("d", r.doctor_id)]
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, ls=ls, lw=lw, zorder=z, solid_capstyle="round")
    for (kind, ident), xy in pos.items():
        member = (kind, ident) in members
        ax.scatter(*xy, s=46 if kind == "d" else 30, marker="s" if kind == "d" else "o", zorder=5,
                   facecolors=INK if member else "white", edgecolors=INK if member else GRAY, linewidths=0.9)
    if highlight is not None:
        ax.annotate("isolated\nfraud claim", xy=highlight, xytext=(highlight[0], highlight[1] + 0.55), ha="center",
                    fontsize=8, color=INK2, arrowprops={"arrowstyle": "-", "color": INK2, "lw": 0.6})
    ax.set_aspect("equal")
    ax.axis("off")


def ring_graph(seed: int = 1) -> None:
    """Isolated versus ring fraud in actual benchmark worlds, with GB-CAG and HGT outcomes."""
    panels = []
    for dens in [0.9, 0.1, 0.9]:
        cfg = GeneratorConfig(ring_shift=0.0, ring_density=dens, seed=seed)
        path = RESULTS_DIR / "cascade" / "predictions" / f"{run_tag(cfg, 0)}.npz"
        if not path.exists():
            return
        panels.append((cfg, generate(cfg), np.load(path)))
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 5.6), gridspec_kw={"hspace": 0.35})
    titles = ["Isolated fraud ($s$ = 0.9 world)", "Ring, sparse block ($s$ = 0.1)", "Ring, dense block ($s$ = 0.9)"]
    for col, (cfg, world, z) in enumerate(panels):
        test = world.split == "test"
        claims = world.claims[test].reset_index(drop=True)
        truth = world.truth[test].reset_index(drop=True)
        assert (claims["claim_id"].to_numpy() == z["claim_id_test"]).all()
        ft = truth["fraud_type"].to_numpy()
        if col == 0:
            iso = np.flatnonzero(ft == "isolated")
            target = claims.iloc[rng.choice(iso)]
            pat, doc = int(target.patient_id), int(target.doctor_id)
        else:
            ring = int(truth.loc[truth.ring_id > 0, "ring_id"].min())
            in_ring = (truth.ring_id == ring).to_numpy()
            ring_docs, ring_pats = set(claims.doctor_id[in_ring]), set(claims.patient_id[in_ring])
        for row, model in enumerate(["GB-CAG", "HGT"]):
            thr = val_threshold(z["y_val"], z[f"{model}__val"])
            flagged = z[f"{model}__test"] >= thr
            pairs = _pair_table(claims, flagged, ft)
            ax = axes[row, col]
            if col == 0:
                pos, edges = _isolated_layout(pairs, pat, doc, np.random.default_rng(1))
                _draw_panel(ax, pos, edges, set(), highlight=(pos[("p", pat)] + pos[("d", doc)]) / 2)
                is_target = (claims.claim_id == target.claim_id).to_numpy()
                note = f"this claim {'flagged' if flagged[is_target][0] else 'missed'}\n" \
                       f"all isolated claims flagged: {flagged[ft == 'isolated'].sum()}/{(ft == 'isolated').sum()}"
            else:
                pos, edges = _ring_layout(pairs, ring_docs, ring_pats, np.random.default_rng(1))
                members = {("d", d) for d in ring_docs} | {("p", p) for p in ring_pats}
                _draw_panel(ax, pos, edges, members)
                note = f"this ring's claims flagged: {flagged[in_ring].sum()}/{in_ring.sum()}\n" \
                       f"legitimate claims flagged (drawn): {edges.legit_flag.sum()}/{edges.legit.sum()}"
            if row == 0:
                ax.set_title(titles[col], fontsize=8, color=INK)
            ax.text(0.5, -0.02, note, transform=ax.transAxes, ha="center", va="top", fontsize=8, color=INK2)
        axes[0, 0].text(-0.08, 0.5, "GB-CAG", transform=axes[0, 0].transAxes, rotation=90, va="center", ha="right",
                        fontsize=8, color=INK)
        axes[1, 0].text(-0.08, 0.5, "HGT", transform=axes[1, 0].transAxes, rotation=90, va="center", ha="right",
                        fontsize=8, color=INK)
    handles = [Line2D([], [], color=CAUGHT, lw=2, label="fraud claims flagged"),
               Line2D([], [], color=MISSED, lw=2, ls=(0, (3, 1.5)), label="fraud claims missed"),
               Line2D([], [], color=FALSE_ALARM, lw=2, ls=(0, (1, 1)), label="legitimate claim flagged"),
               Line2D([], [], color=QUIET, lw=1, label="legitimate, not flagged"),
               Line2D([], [], marker="s", ls="none", mfc=INK, mec=INK, label="ring doctor"),
               Line2D([], [], marker="o", ls="none", mfc=INK, mec=INK, label="ring patient"),
               Line2D([], [], marker="s", ls="none", mfc="white", mec=GRAY, label="other doctor"),
               Line2D([], [], marker="o", ls="none", mfc="white", mec=GRAY, label="other patient")]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.04, right=0.99, top=0.95, bottom=0.08, wspace=0.05)
    fig.savefig(FIG / "fig_ring_graph.pdf", bbox_inches="tight")
    plt.close(fig)


ROUTE_STYLE = {"rerank": ("-", "o", "routed, re-rank"), "fuse": ("--", "s", "routed, fused")}


def cascade_figure(gnn: str = "HGT") -> None:
    """Precision-recall curves and AP against the escalation fraction for GB-CAG -> GNN routing."""
    rows, curves, _ = load_cascade()
    if rows.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.8))
    # (a) PR curves, delta = 0, s = 0.9, mean precision at fixed recall over the six worlds
    grid = np.linspace(0, 1, 101)
    ax = axes[0, 0]
    sel = rows[(rows.ring_shift == 0.0) & (rows.ring_density == 0.9)]
    curves_pr = {k: [] for k in ["GB-CAG", gnn, "rerank", "fuse"]}
    for seed in sorted(sel.seed.unique()):
        z = np.load(RESULTS_DIR / "cascade" / "predictions" / f"emergent_d0.0_s0.9_g{int(seed)}_m0.npz")
        g_val, g_te, h_val, h_te = z["GB-CAG__val"], z["GB-CAG__test"], z[f"{gnn}__val"], z[f"{gnn}__test"]
        scores = {"GB-CAG": g_te, gnn: h_te}
        for mode in ["rerank", "fuse"]:
            e = sel[(sel.seed == seed) & (sel.method == f"GB-CAG>{gnn} ({mode})")]["escalation"].iloc[0]
            scores[mode] = route(g_te, h_te, g_val, h_val, e, mode)[0]
        for k, s in scores.items():
            p, r, _ = precision_recall_curve(z["y_test"], s)    # recall decreasing along the arrays
            p_interp = np.maximum.accumulate(p)                   # best precision at recall >= r
            curves_pr[k].append(np.interp(grid, r[::-1], p_interp[::-1]))
    for k, vals in curves_pr.items():
        color, ls, label = ((BLUE, "-", "GB-CAG") if k == "GB-CAG" else (ORANGE, "-", gnn) if k == gnn
                            else (AQUA, ROUTE_STYLE[k][0], ROUTE_STYLE[k][2]))
        ax.plot(grid, np.mean(vals, 0), color=color, ls=ls, lw=1.6, label=label)
    ax.set_xlabel("recall (all fraud)")
    ax.set_ylabel("precision")
    ax.set_ylim(0, 1.02)
    ax.set_title("(a) precision-recall, $\\delta$ = 0, $s$ = 0.9 (6 worlds)", color=INK)
    ax.legend(frameon=False, loc="lower left", fontsize=8)
    # (b-d) paired difference from GB-CAG against the escalation fraction, all 54 worlds
    keys = ["ring_shift", "ring_density", "seed"]
    for ax, metric, title in zip([axes[0, 1], axes[1, 0], axes[1, 1]], ["ap", "ring_ap", "isolated_ap"],
                                 ["(b) all fraud", "(c) ring fraud", "(d) isolated fraud"]):
        x = np.arange(len(ESCALATION))
        base = rows[rows.method == "GB-CAG"].set_index(keys)[metric]
        alone = rows[rows.method == gnn].set_index(keys)[metric] - base
        # Intervals over the six seeds: a seed's nine regimes share its legitimate claims and ring members.
        lo, hi = t_ci(alone.groupby(level="seed").mean().to_numpy())
        ax.axhline(0.0, color=BLUE, lw=1.2)
        ax.axhspan(lo, hi, color=ORANGE, alpha=0.15, lw=0)
        ax.axhline(alone.mean(), color=ORANGE, lw=1.2)
        box = {"facecolor": "white", "edgecolor": "none", "pad": 0.6}
        ax.text(len(ESCALATION) - 0.55, 0.0, "GB-CAG", color=INK2, fontsize=8, va="center", ha="left", bbox=box)
        ax.text(len(ESCALATION) - 0.55, alone.mean(), f"{gnn} alone", color=INK2, fontsize=8, va="center",
                ha="left", bbox=box)
        for mode in ["rerank", "fuse"]:
            c = curves[(curves.method == f"GB-CAG>{gnn}") & (curves["mode"] == mode)].set_index(keys)
            c = c.assign(diff=c[metric] - base.reindex(c.index).to_numpy())
            g = c.groupby(["escalation", "seed"])["diff"].mean().groupby(level="escalation")
            mean, half = g.mean().reindex(ESCALATION), g.apply(_ci_half).reindex(ESCALATION)
            ls, marker, label = ROUTE_STYLE[mode]
            ax.errorbar(x, mean.values, yerr=half.values, color=AQUA, ls=ls, marker=marker, ms=4, capsize=2,
                        elinewidth=0.8, mec="white", mew=0.6, label=label)
        ax.set_xticks(x, [f"{e:.0%}" for e in ESCALATION])
        ax.set_xlim(-0.4, len(ESCALATION) + 1.0)
        ax.set_xlabel(f"share of claims escalated to {gnn}")
        ax.set_ylabel("$\\Delta$ AP vs. GB-CAG")
        ax.set_title(title + ": paired $\\Delta$ AP (95% CI over 6 seeds)", color=INK)
    axes[0, 1].legend(frameon=False, loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig_cascade.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    ring_graph()
    cascade_figure()
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
    dose = []
    for vol, sub in DOSE_SETS.items():
        d = load_rows(sub)
        if not d.empty:
            dose.append(d[(d.ring_shift == 0.0) & d.seed.isin(ROUND1_SEEDS)].assign(volume=vol))
    if len(dose) > 1:
        dose_response(pd.concat(dose, ignore_index=True))
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
