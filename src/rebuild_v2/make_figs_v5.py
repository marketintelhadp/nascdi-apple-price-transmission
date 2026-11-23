import sys
from pathlib import Path
import numpy as np, pandas as pd, statsmodels.api as sm
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.dates as mdates
from PIL import Image
import sys as _sys; from pathlib import Path; _sys.path.insert(0, str(Path(__file__).resolve().parent))
from regen import *

OUT = REPO / "results_rebuild_v2" / "figures"; OUT.mkdir(exist_ok=True); OLD = REPO / "data" / "figures_original"
RED, NAVY = "#B03A2E", "#1F3A5F"
SH = {c: f"{c.split('-')[0][:3]}-{c.split('-')[1][:3]}-{c.split('-')[2]}" for c in CHAINS}
col = lambda c: RED if c.startswith("Shopian") else NAVY
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
def fig_px(w, h, dpi=220): return plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)

# ---------------- Figure 4: weekly index (upper) and weekly record count (lower)
w = pd.read_csv(REPO / "data/nascdi_news_v2/nascdi_news_weekly.csv", parse_dates=["week_end"]).set_index("week_end")
f, ax = plt.subplots(2, 1, figsize=(9, 6), dpi=200, sharex=True, gridspec_kw={"hspace": 0.08})
ax[0].plot(w.index, w.NASCDI, color="#1F3A5F", lw=0.8); ax[0].set_ylabel("Weekly NASCDI")
ax[0].axvspan(pd.Timestamp("2025-06-15"), pd.Timestamp("2025-07-01"), color="0.85", lw=0)
cnt = w.news_volume.where(w.news_volume > 0); ax[1].plot(w.index, cnt, color="#2E7D32", lw=0.8); ax[1].set_yscale("log"); ax[1].set_ylabel("Qualifying records per week (log scale)")
ax[1].axvspan(pd.Timestamp("2025-06-15"), pd.Timestamp("2025-07-01"), color="0.85", lw=0)
for a in ax: a.axvline(pd.Timestamp("2020-10-04"), color="0.45", ls=":", lw=1)
ax[0].text(pd.Timestamp("2025-06-20"), ax[0].get_ylim()[1] * 0.96, "no GKG records\n15 Jun-1 Jul 2025", ha="right", va="top", fontsize=7, color="0.35")
f.savefig(OUT / "fig4.png", bbox_inches="tight"); plt.close(f)

# ---------------- Figure 3: keep panels 1-2 of the original, replace the NASCDI panel
im = Image.open(OLD / "fig3.jpeg").convert("RGB"); W, H = im.size
top = im.crop((0, 0, W, 635))
nw = [pd.Timestamp(x) for x in sorted(set().union(*[set(CH[c].week_end) for c in CHAINS]))]; grid = pd.date_range(min(nw), max(nw), freq="W-SUN")
fw, fh = W / 220, (H - 635) / 220
f = plt.figure(figsize=(fw, fh), dpi=220); ax = f.add_axes([78 / W, 0.30, (1303 - 78) / W, 0.66])
ax.plot(IDX.reindex(grid).index, IDX.reindex(grid).NASCDI, color="black", lw=0.6); ax.set_ylabel("NASCDI", fontsize=6.2); ax.yaxis.set_label_coords(-0.036, 0.5); ax.set_xlabel("Week", fontsize=6.2); ax.tick_params(labelsize=5.6, width=0.6, length=2.5)
x0 = pd.Timestamp("2016-01-01") - pd.Timedelta(days=(170 - 78) / 118.6 * 365.25); x1 = pd.Timestamp("2016-01-01") + pd.Timedelta(days=(1303 - 170) / 118.6 * 365.25); ax.set_xlim(x0, x1)
ax.xaxis.set_major_locator(mdates.YearLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y")); [sp.set_linewidth(0.6) for sp in ax.spines.values()]; ax.spines['top'].set_visible(True); ax.spines['right'].set_visible(True)
f.savefig(OUT / "_panel3.png", dpi=220); plt.close(f)
p3 = Image.open(OUT / "_panel3.png").convert("RGB").resize((W, H - 635)); new = Image.new("RGB", (W, H), "white"); new.paste(top, (0, 0)); new.paste(p3, (0, 635)); new.save(OUT / "fig3.jpeg", quality=92)

# ---------------- Figure 7: ECT with HAC intervals;  Figure 8: bounds F
t6 = pd.read_csv(TAB / "table6_8_baseline.csv"); order = list(t6.chain)
f = fig_px(1364, 792); ax = f.add_subplot(111)
for i, r in t6.iterrows():
    y = len(t6) - 1 - i; ax.errorbar(r.ECT, y, xerr=[[r.ECT - r.ECT_lo], [r.ECT_hi - r.ECT]], fmt="o", color=col(r.chain), capsize=3, lw=1.5)
ax.set_yticks(range(len(t6))); ax.set_yticklabels([SH[c] for c in order][::-1]); ax.axvline(0, color="black", lw=0.8); ax.set_xlabel("Error-correction coefficient (weekly), 95% HAC interval"); ax.set_title("Shopian (red) and Sopore (blue) chains", fontsize=8)
f.tight_layout(); f.savefig(OUT / "fig7.png"); plt.close(f)
f = fig_px(1364, 792); ax = f.add_subplot(111)
for i, r in t6.iterrows():
    ax.bar(i, r.F, width=0.6, color=col(r.chain), alpha=0.85); ax.hlines(r.F_hi, i - 0.4, i + 0.4, color="black", lw=1.4); ax.hlines(r.F_lo, i - 0.4, i + 0.4, color="black", lw=1.2, ls="--")
ax.set_xticks(range(len(t6))); ax.set_xticklabels([SH[c] for c in order], rotation=35, ha="right"); ax.set_ylabel("Bounds F statistic"); ax.set_title("Solid line: simulated 5% upper bound; dashed: 5% lower bound", fontsize=8)
f.tight_layout(); f.savefig(OUT / "fig8.png"); plt.close(f)

# ---------------- Figure 9: partial sums
f = fig_px(1364, 748); ax = f.add_subplot(111); ix = IDX.index
ax.plot(ix, IDX.P, color=RED, lw=1.6, label="NASCDI⁺ (cumulative increases)"); ax.plot(ix, -IDX.N, color=NAVY, lw=1.6, label="−NASCDI⁻ (cumulative decreases, sign reversed)")
tt = np.arange(len(ix)); b = np.polyfit(tt, IDX.P.values, 1); ax.plot(ix, np.polyval(b, tt), color="black", ls=":", lw=1, label="linear time trend fitted to NASCDI⁺")
ax.set_ylabel("Partial sum (index points)"); ax.legend(frameon=False, fontsize=7); f.tight_layout(); f.savefig(OUT / "fig9.png"); plt.close(f)

# ---------------- Figure 10: bootstrap p-values for symmetry
f = fig_px(1364, 792); ax = f.add_subplot(111)
for i, r in t6.iterrows():
    y = len(t6) - 1 - i
    ax.plot(r.LR_pboot, y, "o", color="0.2", ms=4, label="long-run" if i == 0 else None); ax.plot(r.SR_pboot, y + 0.0, "s", color="0.2", ms=4, label="short-run" if i == 0 else None); ax.plot(r.J_pboot, y + 0.12, "^", color="0.2", ms=4, label="joint" if i == 0 else None)
ax.set_xscale("log"); ax.axvline(0.05, color=RED, ls="--", lw=0.9); ax.set_yticks(range(len(t6))); ax.set_yticklabels([SH[c] for c in order][::-1]); ax.set_xlabel("Wild-bootstrap p-value for symmetry (log scale; dashed = 0.05)"); ax.legend(frameon=False, fontsize=7, loc="lower right")
f.tight_layout(); f.savefig(OUT / "fig10.png"); plt.close(f)

# ---------------- Figure 11: dynamic responses (one-week spike, 90th percentile = 11.4 points), 90% bands
pct = pd.read_csv(TAB / "tableA8b_percentiles.csv").set_index("percentile")["increase"]; s = float(pct[90]); H_ = 26
def irf(params, names, p, r, s, h):
    g = lambda nm: params[names.index(nm)] if nm in names else 0.0
    lam, gam = g("y_L1"), g("a_L1"); phi = [g(f"dy_L{i}") for i in range(1, p + 1)]; psi = [g(f"da_L{i}") for i in range(r + 1)]
    T = H_ + 1; a = np.zeros(T + 1); a[1:h + 1] = s; y = np.zeros(T + 1); dy = np.zeros(T + 1)
    for t in range(1, T + 1):
        da = lambda k: (a[t - k] - a[t - k - 1]) if t - k >= 1 else 0.0
        e = lam * y[t - 1] + gam * a[t - 1] + sum(phi[i - 1] * (dy[t - i] if t - i >= 1 else 0.0) for i in range(1, p + 1)) + sum(psi[i] * da(i) for i in range(r + 1))
        dy[t] = e; y[t] = y[t - 1] + e
    return 100 * y[1:]
f, axs = plt.subplots(2, 4, figsize=(1980 / 220, 968 / 220), dpi=220, sharex=True)
for ax, ch in zip(axs.ravel(), CHAINS):
    o, (d, dd, X, m, (p, q, r)) = run(CH[ch], IDX, "prod", "level", False, with_sym=False); names = list(X.columns)
    hac = sm.OLS(dd["dy"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 4, "use_correction": True}); b = hac.params.values; V = hac.cov_params().values
    dr = rng.multivariate_normal(b, V, size=4000); li = names.index("y_L1"); dr = dr[(dr[:, li] < 0) & (dr[:, li] > -1)][:2000]
    pt = irf(b, names, p, r, s, 1); D = np.array([irf(x, names, p, r, s, 1) for x in dr]); lo, hi = np.percentile(D, [5, 95], axis=0); wk = np.arange(1, len(pt) + 1)
    ax.fill_between(wk, lo, hi, color=col(ch), alpha=0.18, lw=0); ax.plot(wk, pt, color=col(ch), lw=1.6); ax.axhline(0, color="black", lw=0.6); ax.set_title(SH[ch], fontsize=8)
for ax in axs[1]: ax.set_xlabel("weeks")
for ax in axs[:, 0]: ax.set_ylabel("% price response")
f.suptitle("Response of producer price to a one-week NASCDI spike of %.1f points (90th percentile weekly increase); 90%% bands" % s, fontsize=8); f.tight_layout(rect=[0, 0, 1, 0.95]); f.savefig(OUT / "fig11.png"); plt.close(f)

# ---------------- Figures 12 and 13: recursive CUSUM and CUSUMSQ
f12, a12 = plt.subplots(2, 4, figsize=(1540 / 220, 858 / 220), dpi=220); f13, a13 = plt.subplots(2, 4, figsize=(1540 / 220, 858 / 220), dpi=220)
for k, ch in enumerate(CHAINS):
    o, (d, dd, X, m, _) = run(CH[ch], IDX, "prod", "nardl", False, with_sym=False); Xv, yv = X.values, dd["dy"].values; n, kk = Xv.shape; wr = []
    for i in range(kk, n):
        b = np.linalg.lstsq(Xv[:i], yv[:i], rcond=None)[0]; A = np.linalg.pinv(Xv[:i].T @ Xv[:i]); wr.append((yv[i] - Xv[i] @ b) / np.sqrt(1 + Xv[i] @ A @ Xv[i]))
    wr = np.array(wr); T = len(wr); j = np.arange(1, T + 1); W_ = np.cumsum(wr) / wr.std(ddof=1); bd = 0.948 * (np.sqrt(T) + 2 * j / np.sqrt(T))
    ax = a12.ravel()[k]; ax.plot(j, W_, color=col(ch), lw=1); ax.plot(j, bd, "--", color="0.35", lw=0.9); ax.plot(j, -bd, "--", color="0.35", lw=0.9); ax.axhline(0, color="black", lw=0.6); ax.set_title(f"{SH[ch]} (T = {T})", fontsize=8)
    S = np.cumsum(wr ** 2) / (wr ** 2).sum(); c0 = 1.358 / np.sqrt(T / 2); ax = a13.ravel()[k]; ax.plot(j, S, color=col(ch), lw=1); ax.plot(j, j / T, color="black", lw=0.6)
    ax.plot(j, np.clip(j / T + c0, 0, 1), "--", color="0.35", lw=0.9); ax.plot(j, np.clip(j / T - c0, 0, 1), "--", color="0.35", lw=0.9); ax.set_title(f"{SH[ch]} (T = {T})", fontsize=8)
for F_ in (f12, f13):
    for ax in F_.axes: ax.tick_params(labelsize=7)
    F_.supxlabel("Observation (recursive residuals)", fontsize=8); F_.tight_layout()
f12.savefig(OUT / "fig12.png"); f13.savefig(OUT / "fig13.png"); plt.close("all")

# ---------------- Figure 14: bootstrap p-values of the bounds t-test (NARDL and level specifications)
A9 = pd.read_csv(TAB.parent / "tables_new" / "A9.csv")
f, axs = plt.subplots(1, 2, figsize=(1540 / 220, 682 / 220), dpi=220, sharey=True)
for ax, spec, ttl in zip(axs, ("nardl", "level"), ("(a) NARDL specification (NASCDI⁺, NASCDI⁻)", "(b) NASCDI entered as a level")):
    for i, ch in enumerate(CHAINS):
        for dep, mk, c, off in (("prod", "o", NAVY, -0.12), ("term", "s", RED, 0.12)):
            v = float(A9[(A9.chain == ch) & (A9.dep == dep) & (A9.spec == spec)].pt.iloc[0]); ax.plot(i + off, max(v, 0.0005), mk, color=c, ms=4, label=("Producer price dependent (baseline)" if dep == "prod" else "Terminal price dependent") if i == 0 else None)
    ax.set_yscale("log"); ax.axhline(0.05, color="black", ls="--", lw=0.9); ax.set_xticks(range(8)); ax.set_xticklabels([SH[c] for c in CHAINS], rotation=60, ha="right", fontsize=7); ax.set_title(ttl, fontsize=8); ax.set_ylim(0.0003, 1)
axs[0].set_ylabel("Bootstrap p-value of the bounds t-test"); axs[0].legend(frameon=False, fontsize=6, loc="lower left"); f.tight_layout(); f.savefig(OUT / "fig14.png"); plt.close(f)
print("figures written:", sorted(p.name for p in OUT.glob("*.*")))
