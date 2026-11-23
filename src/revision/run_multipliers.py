"""Referee 1 #2 and Referee 2 minor: dynamic responses to EMPIRICALLY representative NASCDI shocks,
with parametric-bootstrap bands (HAC covariance), from the symmetric ARDL/ECM of each chain."""
import numpy as np
import pandas as pd

from . import rev_lib as L

H = 20
SHOCKS = {"1 SD of weekly change (5.6)": 5.58, "90th pct weekly increase (15.6)": 15.58, "99th pct weekly increase (25.9)": 25.91}


def respond(params: pd.Series, p: int, r: int, zpath: np.ndarray):
    a = params["y_L1"]
    g = params["Z_L1"]
    gam = [params.get(f"dy_L{i}", 0.0) for i in range(1, p + 1)]
    th = [params.get(f"dZ_L{k}", 0.0) for k in range(0, r + 1)]
    T = len(zpath)
    ydev = np.zeros(T + 1)  # index shift: ydev[t+1] = level at t ; ydev[0] = level at -1
    dy = np.zeros(T)
    zprev = 0.0
    dz = np.zeros(T)
    for t in range(T):
        dz[t] = zpath[t] - (zpath[t - 1] if t > 0 else 0.0)
    for t in range(T):
        zlag = zpath[t - 1] if t > 0 else 0.0
        v = a * ydev[t] + g * zlag
        for i, gi in enumerate(gam, 1):
            if t - i >= 0:
                v += gi * dy[t - i]
        for k, tk in enumerate(th):
            if t - k >= 0:
                v += tk * dz[t - k]
        dy[t] = v
        ydev[t + 1] = ydev[t] + v
    return ydev[1:]


if __name__ == "__main__":
    base = pd.read_csv(L.OUT_DIR / "main_base.csv").set_index("chain")
    nas = L.load_nascdi()
    rng = np.random.default_rng(31)
    rows = []
    for name, path in L.chains():
        p, q, r = int(base.loc[name, "p"]), int(base.loc[name, "q"]), int(base.loc[name, "r"])
        f = L.chain_frame(path, nas["NASCDI"])
        d, y, X = L.build_design(f, p, q, r, "sym")
        hac = L.ols(y, X, "HAC")
        b, V = hac.params, hac.cov_params()
        draws = rng.multivariate_normal(b.values, V.values, size=2000)
        for sname, s in SHOCKS.items():
            for kind in ("pulse (1 week)", "sustained (4 weeks)"):
                z = np.zeros(H + 1)
                if kind.startswith("pulse"):
                    z[0] = s
                else:
                    z[:4] = s
                pt = respond(b, p, r, z) * 100  # log points -> approx %
                sims = np.array([respond(pd.Series(dr, index=b.index), p, r, z) * 100 for dr in draws])
                lo, hi = np.percentile(sims, [5, 95], axis=0)
                pk = int(np.argmax(np.abs(pt)))
                rows.append(dict(chain=name, shock=sname, kind=kind, peak_pct=pt[pk], peak_week=pk, peak_lo90=lo[pk], peak_hi90=hi[pk],
                                 wk4=pt[4], wk8=pt[8], wk20=pt[20], wk20_lo90=lo[20], wk20_hi90=hi[20]))
        # per-index-point long-run effect (for comparison with the submitted Figure 10)
        lr = -b["Z_L1"] / b["y_L1"]
        se = np.std([-dr[list(b.index).index("Z_L1")] / dr[list(b.index).index("y_L1")] for dr in draws])
        rows.append(dict(chain=name, shock="long-run effect per index point (%)", kind="permanent",
                         peak_pct=100 * lr, peak_lo90=100 * (lr - 1.645 * se), peak_hi90=100 * (lr + 1.645 * se)))
    out = pd.DataFrame(rows)
    out.to_csv(L.OUT_DIR / "dynamic_responses_representative_shocks.csv", index=False)
    pd.set_option("display.width", 250)
    print(out[out.kind != "permanent"].groupby(["shock", "kind"])[["peak_pct", "wk20"]].agg(["min", "max"]).round(2).to_string())
