"""Figures for the revised manuscript (replace Figs 6-13; add coverage and partial-sum figures)."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import rev_lib as L
from .run_multipliers import respond

FIG = L.OUT_DIR / "figs"
FIG.mkdir(exist_ok=True)
C = {"Shopian": "#B03A2E", "Sopore": "#1F3A5F"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def short(c):
    m, v, g = c.split("-")
    return f"{m[:3]}-{v[:3]}-{g}"


def fig_ect(d):
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    y = np.arange(len(d))[::-1]
    for yi, (_, r) in zip(y, d.iterrows()):
        col = C[r.chain.split("-")[0]]
        ax.errorbar(r.ECT, yi, xerr=[[r.ECT - r.ECT_lo], [r.ECT_hi - r.ECT]], fmt="o", color=col, capsize=3)
    ax.axvline(0, color="k", lw=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels([short(c) for c in d.chain])
    ax.set_xlabel("Error-correction coefficient (weekly), 95% HAC interval")
    ax.set_title("Shopian (red) and Sopore (blue) chains", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig6_ect_ci.png", dpi=220)
    plt.close(fig)


def fig_bounds(d):
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    x = np.arange(len(d))
    for xi, (_, r) in zip(x, d.iterrows()):
        col = C[r.chain.split("-")[0]]
        ax.bar(xi, r.F, color=col, alpha=0.85, width=0.6)
        ax.hlines(r.F_U5, xi - 0.4, xi + 0.4, color="k", lw=1.2)
        ax.hlines(r.F_L5, xi - 0.4, xi + 0.4, color="k", lw=1.2, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([short(c) for c in d.chain], rotation=35, ha="right")
    ax.set_ylabel("Bounds F statistic")
    ax.set_title("Solid line: simulated 5% upper bound; dashed: 5% lower bound", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig7_bounds_F.png", dpi=220)
    plt.close(fig)


def fig_sym(d):
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    y = np.arange(len(d))[::-1]
    off = {"LR": -0.2, "SR": 0.0, "JT": 0.2}
    mk = {"LR": "o", "SR": "s", "JT": "^"}
    for t in ("LR", "SR", "JT"):
        ax.scatter(d[f"{t}_p_boot"], y + off[t], marker=mk[t], label={"LR": "long-run", "SR": "short-run", "JT": "joint"}[t],
                   color="#333333", s=22)
    ax.axvline(0.05, color="#B03A2E", ls="--", lw=0.8)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels([short(c) for c in d.chain])
    ax.set_xlabel("Wild-bootstrap p-value for symmetry (log scale; dashed = 0.05)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "fig8_symmetry_p.png", dpi=220)
    plt.close(fig)


def fig_partial_sums(nas):
    z = nas["NASCDI"].loc[:"2025-03-02"]
    P, N = L.partial_sums(z)
    t = np.arange(len(z))
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.plot(z.index, P, color="#B03A2E", label="NASCDI⁺ (cumulative increases)")
    ax.plot(z.index, -N, color="#1F3A5F", label="−NASCDI⁻ (cumulative decreases, sign reversed)")
    ax.plot(z.index, np.polyval(np.polyfit(t, P.values, 1), t), color="k", ls=":", lw=0.9, label="linear time trend fitted to NASCDI⁺")
    ax.set_ylabel("Partial sum (index points)")
    ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig_partial_sums.png", dpi=220)
    plt.close(fig)


def fig_multipliers(base, nas):
    rng = np.random.default_rng(31)
    s = 15.58
    fig, axes = plt.subplots(2, 4, figsize=(9, 4.4), sharex=True)
    for ax, (name, path) in zip(axes.ravel(), L.chains()):
        p, q, r = int(base.loc[name, "p"]), int(base.loc[name, "q"]), int(base.loc[name, "r"])
        f = L.chain_frame(path, nas["NASCDI"])
        d, y, X = L.build_design(f, p, q, r, "sym")
        hac = L.ols(y, X, "HAC")
        b, V = hac.params, hac.cov_params()
        z = np.zeros(21)
        z[0] = s
        pt = respond(b, p, r, z) * 100
        sims = np.array([respond(pd.Series(dr, index=b.index), p, r, z) * 100
                         for dr in rng.multivariate_normal(b.values, V.values, size=1500)])
        lo, hi = np.percentile(sims, [5, 95], axis=0)
        h = np.arange(21)
        ax.fill_between(h, lo, hi, color=C[name.split("-")[0]], alpha=0.2)
        ax.plot(h, pt, color=C[name.split("-")[0]])
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(short(name), fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("weeks")
    axes[0, 0].set_ylabel("% price response")
    axes[1, 0].set_ylabel("% price response")
    fig.suptitle("Response of producer price to a one-week NASCDI spike of 15.6 points (90th percentile weekly increase); 90% bands", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig10_multipliers.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    base = pd.read_csv(L.OUT_DIR / "main_base.csv")
    nas = L.load_nascdi()
    fig_ect(base)
    fig_bounds(base)
    fig_sym(base)
    fig_partial_sums(nas)
    fig_multipliers(base.set_index("chain"), nas)
    print("figures written to", FIG)
