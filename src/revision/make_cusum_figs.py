"""Figures 12-13: recursive CUSUM and CUSUMSQ paths for the revised unrestricted NARDL equations (same models as Table 9)."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import rev_lib as L
from .run_cusum import recursive_resid
from .make_figures import C, short, FIG

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})


def paths():
    base = pd.read_csv(L.OUT_DIR / "main_base.csv").set_index("chain")
    nas = L.load_nascdi()
    out = {}
    for name, path in L.chains():
        p, q, r = int(base.loc[name, "p"]), int(base.loc[name, "q"]), int(base.loc[name, "r"])
        f = L.chain_frame(path, nas["NASCDI"])
        d, y, X = L.build_design(f, p, q, r, "asym")
        w = recursive_resid(y.values, X.values)
        T = len(w)
        t = np.arange(1, T + 1)
        W = np.cumsum(w) / np.sqrt(np.sum(w ** 2) / T)
        bound = 0.948 * np.sqrt(T) * (1 + 2 * t / T)
        s = np.cumsum(w ** 2) / np.sum(w ** 2)
        c0 = 1.358 / np.sqrt(T / 2.0)
        out[name] = dict(T=T, t=t, W=W, bound=bound, s=s, line=t / T, c0=c0)
    return out


def draw(P, kind, fname):
    fig, axes = plt.subplots(2, 4, figsize=(7.0, 3.9), sharex=False)
    for ax, (name, v) in zip(axes.ravel(), P.items()):
        col = C[name.split("-")[0]]
        if kind == "cusum":
            ax.plot(v["t"], v["W"], color=col, lw=1.1)
            ax.plot(v["t"], v["bound"], color="0.35", lw=0.8, ls="--")
            ax.plot(v["t"], -v["bound"], color="0.35", lw=0.8, ls="--")
            ax.axhline(0, color="k", lw=0.5)
        else:
            ax.plot(v["t"], v["s"], color=col, lw=1.1)
            ax.plot(v["t"], v["line"], color="k", lw=0.6)
            ax.plot(v["t"], np.clip(v["line"] + v["c0"], 0, 1), color="0.35", lw=0.8, ls="--")
            ax.plot(v["t"], np.clip(v["line"] - v["c0"], 0, 1), color="0.35", lw=0.8, ls="--")
        ax.set_title(f"{short(name)} (T = {v['T']})", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.supxlabel("Observation (recursive residuals)", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    P = paths()
    draw(P, "cusum", "fig12_cusum.png")
    draw(P, "cusumsq", "fig13_cusumsq.png")
    b = pd.read_csv(L.OUT_DIR / "cusum_revised.csv").set_index("chain")
    for n, v in P.items():
        print(n, "CUSUM max |W|/bound %.2f" % np.max(np.abs(v["W"]) / v["bound"]), "| CUSUMSQ max dev %.3f band %.3f" % (np.max(np.abs(v["s"] - v["line"])), v["c0"]),
              "| csv breach:", b.loc[n, "cusum_breach_BDE"], b.loc[n, "cusumsq_breach"])
