"""Figure 14: bootstrap p-values of the bounds t-test, producer price vs terminal price as the dependent variable."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import rev_lib as L
from .make_figures import short, FIG

AUD = L.OUT_DIR / "audit"
fw = pd.read_csv(AUD / "bounds_bootstrap_producer.csv")
rv = pd.read_csv(AUD / "bounds_bootstrap_reverse.csv")
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1), sharey=True)
titles = {"asym": "(a) NARDL specification (NASCDI⁺, NASCDI⁻)", "sym": "(b) NASCDI entered as a level"}
for ax, spec in zip(axes, ("asym", "sym")):
    a = fw[fw.spec == spec].set_index("chain"); b = rv[rv.spec == spec].set_index("chain")
    chains = list(a.index); x = np.arange(len(chains))
    ax.scatter(x - 0.1, a.loc[chains, "pt_boot"], marker="o", s=26, color="#1F3A5F", label="Producer price dependent (baseline)", zorder=3)
    ax.scatter(x + 0.1, b.loc[chains, "pt_boot"], marker="s", s=26, color="#B03A2E", label="Terminal price dependent", zorder=3)
    ax.axhline(0.05, color="k", lw=0.8, ls="--")
    ax.set_yscale("log"); ax.set_ylim(0.002, 1.0)
    ax.set_xticks(x); ax.set_xticklabels([short(c) for c in chains], rotation=60, ha="right", fontsize=7)
    ax.set_title(titles[spec], fontsize=8)
axes[0].set_ylabel("Bootstrap p-value of the bounds t-test")
axes[0].legend(loc="lower left", fontsize=6.5, frameon=False)
fig.tight_layout()
fig.savefig(FIG / "fig14_direction.png", dpi=220)
