#!/usr/bin/env python3
"""
Structural Breaks + Sensitivity Analysis for Vulnerability Index (VI)
Author: (your name)
Usage:
  python structural_breaks_sensitivity_vi.py --input path/to/vi.csv --outdir results_vi

Expected input CSV columns:
  - date column (default: 'date') parseable by pandas
  - VI column (default: 'VI') numeric in [0,1] (or any continuous scale)
Optional (for sensitivity):
  - component columns (e.g., Exposure, Sensitivity, AdaptiveCapacity, ...)
  - weights can be passed via JSON string or a weights file

Outputs:
  - breakpoints table + regime summary CSV
  - sensitivity summary CSV
  - plots (break overlay, regime means, variance breaks)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Utilities
# -----------------------------
def ensure_outdir(outdir: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    return outdir


def safe_float_series(s: pd.Series, name: str) -> pd.Series:
    s2 = pd.to_numeric(s, errors="coerce")
    if s2.isna().mean() > 0.2:
        raise ValueError(f"Too many NaNs after coercing {name} to numeric. Check your input column.")
    return s2


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    if not weights:
        raise ValueError("Weights dict is empty.")
    s = float(sum(weights.values()))
    if s <= 0:
        raise ValueError("Sum of weights must be positive.")
    return {k: float(v) / s for k, v in weights.items()}


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return pd.Series(a).corr(pd.Series(b), method="spearman")


def kendall(a: np.ndarray, b: np.ndarray) -> float:
    return pd.Series(a).corr(pd.Series(b), method="kendall")


def classification_stability(base: np.ndarray, alt: np.ndarray, thr: float) -> float:
    base_c = (base >= thr).astype(int)
    alt_c = (alt >= thr).astype(int)
    return float((base_c == alt_c).mean())


def robust_zscore(x: np.ndarray) -> np.ndarray:
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if mad == 0:
        return (x - med)  # fallback
    return (x - med) / (1.4826 * mad)


def minmax(x: np.ndarray) -> np.ndarray:
    mn = np.nanmin(x)
    mx = np.nanmax(x)
    if mx - mn == 0:
        return x - mn
    return (x - mn) / (mx - mn)


def zscore(x: np.ndarray) -> np.ndarray:
    mu = np.nanmean(x)
    sd = np.nanstd(x, ddof=1)
    if sd == 0:
        return x - mu
    return (x - mu) / sd


# -----------------------------
# Change-point detection (ruptures if available)
# -----------------------------
def try_import_ruptures():
    try:
        import ruptures as rpt  # type: ignore
        return rpt
    except Exception:
        return None


def pettitt_test(x: np.ndarray) -> Tuple[int, float]:
    """
    Pettitt test for one change-point (non-parametric).
    Returns (cp_index, approx_pvalue).
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    # Rank-based approach:
    r = pd.Series(x).rank().to_numpy()
    U = np.zeros(n)
    for t in range(n):
        U[t] = 2 * np.sum(r[: t + 1]) - (t + 1) * (n + 1)
    K = np.max(np.abs(U))
    cp = int(np.argmax(np.abs(U)))
    # Approx p-value (Pettitt approximation):
    p = 2 * np.exp((-6 * K**2) / (n**3 + n**2))
    p = float(min(max(p, 0.0), 1.0))
    return cp, p


# -----------------------------
# Core: Structural Breaks
# -----------------------------
@dataclass
class BreakResult:
    indices: List[int]
    dates: List[pd.Timestamp]
    method: str


def detect_breaks(
    y: np.ndarray,
    dates: pd.Series,
    method: str = "pelt_rbf",
    pen: float = 3.0,
    min_size: int = 8,
) -> BreakResult:
    rpt = try_import_ruptures()
    n = len(y)
    if n < 3 * min_size:
        return BreakResult(indices=[], dates=[], method=f"{method}_skipped_small_n")

    if rpt is None:
        # Fallback: Pettitt single break only
        cp, p = pettitt_test(y)
        idx = [cp] if (1 <= cp <= n - 2) else []
        dts = [pd.to_datetime(dates.iloc[i])] if idx else []
        return BreakResult(indices=idx, dates=dts, method=f"pettitt_fallback(p≈{p:.3f})")

    y2 = y.reshape(-1, 1)

    if method == "pelt_rbf":
        algo = rpt.Pelt(model="rbf", min_size=min_size).fit(y2)
        bkpts = algo.predict(pen=pen)  # includes n
        idx = [b for b in bkpts if b < n]
        dts = [pd.to_datetime(dates.iloc[i]) for i in idx]
        return BreakResult(indices=idx, dates=dts, method=f"pelt_rbf(pen={pen})")

    if method == "binseg_l2":
        algo = rpt.Binseg(model="l2", min_size=min_size).fit(y2)
        # You can tune "n_bkps"
        n_bkps = max(1, min(5, n // (3 * min_size)))
        bkpts = algo.predict(n_bkps=n_bkps)
        idx = [b for b in bkpts if b < n]
        dts = [pd.to_datetime(dates.iloc[i]) for i in idx]
        return BreakResult(indices=idx, dates=dts, method=f"binseg_l2(n_bkps={n_bkps})")

    raise ValueError(f"Unknown method: {method}")


def summarize_regimes(df: pd.DataFrame, break_indices: List[int], date_col: str, vi_col: str) -> pd.DataFrame:
    idx = [0] + sorted(break_indices) + [len(df)]
    out = []
    for k in range(len(idx) - 1):
        a, b = idx[k], idx[k + 1]
        seg = df.iloc[a:b]
        out.append(
            {
                "regime": k + 1,
                "start": seg[date_col].iloc[0],
                "end": seg[date_col].iloc[-1],
                "n": int(len(seg)),
                "VI_mean": float(seg[vi_col].mean()),
                "VI_sd": float(seg[vi_col].std(ddof=1)),
                "VI_median": float(seg[vi_col].median()),
                "VI_iqr": float(seg[vi_col].quantile(0.75) - seg[vi_col].quantile(0.25)),
                "VI_min": float(seg[vi_col].min()),
                "VI_max": float(seg[vi_col].max()),
            }
        )
    return pd.DataFrame(out)


def plot_breaks(df: pd.DataFrame, date_col: str, vi_col: str, breaks: BreakResult, outpath: str) -> None:
    plt.figure()
    plt.plot(df[date_col], df[vi_col], label="VI")
    for d in breaks.dates:
        plt.axvline(d, linestyle="--")
    plt.title(f"VI with structural breaks ({breaks.method})")
    plt.xlabel("Date")
    plt.ylabel("Vulnerability Index (VI)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


# -----------------------------
# Sensitivity Analysis
# -----------------------------
def compute_vi_from_components(df: pd.DataFrame, components: List[str], weights: Dict[str, float]) -> np.ndarray:
    w = normalize_weights(weights)
    X = df[components].astype(float).to_numpy()
    ww = np.array([w[c] for c in components], dtype=float)
    return X @ ww


def monte_carlo_weight_sensitivity(
    df: pd.DataFrame,
    components: List[str],
    weights: Dict[str, float],
    pct: float,
    draws: int,
    thresholds: List[float],
    seed: int = 7,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    weights = normalize_weights(weights)

    base = compute_vi_from_components(df, components, weights)
    w0 = np.array([weights[c] for c in components], dtype=float)

    rows = []
    for _ in range(draws):
        mult = rng.uniform(1 - pct, 1 + pct, size=len(components))
        w1 = w0 * mult
        w1 = w1 / w1.sum()
        wdict = {c: float(w1[i]) for i, c in enumerate(components)}
        alt = compute_vi_from_components(df, components, wdict)

        row = {
            "pct": pct,
            "spearman": spearman(base, alt),
            "kendall": kendall(base, alt),
            "rmse": float(np.sqrt(np.mean((alt - base) ** 2))),
            "mad": float(np.mean(np.abs(alt - base))),
        }
        for thr in thresholds:
            row[f"class_stability_thr_{thr:.2f}"] = classification_stability(base, alt, thr)
        rows.append(row)

    out = pd.DataFrame(rows)
    # Summarize as quantiles
    summary = out.quantile([0.05, 0.50, 0.95]).T.reset_index()
    summary.columns = ["metric", "p05", "p50", "p95"]
    summary.insert(0, "scenario", f"MC_weights_±{int(pct*100)}%")
    return summary


def leave_one_out_sensitivity(
    df: pd.DataFrame,
    components: List[str],
    weights: Dict[str, float],
) -> pd.DataFrame:
    weights = normalize_weights(weights)
    base = compute_vi_from_components(df, components, weights)
    out = []
    for drop in components:
        comps2 = [c for c in components if c != drop]
        w2 = {c: weights[c] for c in comps2}
        alt = compute_vi_from_components(df, comps2, w2)
        out.append(
            {
                "dropped_component": drop,
                "spearman": spearman(base, alt),
                "kendall": kendall(base, alt),
                "rmse": float(np.sqrt(np.mean((alt - base) ** 2))),
                "mad": float(np.mean(np.abs(alt - base))),
            }
        )
    return pd.DataFrame(out).sort_values("rmse", ascending=False)


def normalization_sensitivity(
    df: pd.DataFrame,
    components: List[str],
    weights: Dict[str, float],
    norm_method: str,
) -> np.ndarray:
    """
    Rebuild VI after applying a chosen normalization to each component.
    """
    weights = normalize_weights(weights)
    X = df[components].astype(float).to_numpy()

    Xn = np.zeros_like(X)
    for j in range(X.shape[1]):
        col = X[:, j]
        if norm_method == "minmax":
            Xn[:, j] = minmax(col)
        elif norm_method == "zscore":
            Xn[:, j] = zscore(col)
        elif norm_method == "robust":
            Xn[:, j] = robust_zscore(col)
        else:
            raise ValueError("norm_method must be one of: minmax, zscore, robust")

    ww = np.array([weights[c] for c in components], dtype=float)
    return Xn @ ww


# -----------------------------
# Main
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to CSV containing date + VI (and optionally components).")
    p.add_argument("--outdir", default="results_vi", help="Output directory.")
    p.add_argument("--date_col", default="date", help="Name of date column.")
    p.add_argument("--vi_col", default="VI", help="Name of vulnerability index column.")
    p.add_argument("--freq", default=None, help="Optional: enforce frequency by resampling (e.g., 'W', 'M').")
    p.add_argument("--break_pen", type=float, default=3.0, help="Penalty for PELT (higher -> fewer breaks).")
    p.add_argument("--min_size", type=int, default=8, help="Minimum segment length for break detection.")
    p.add_argument("--components", default=None,
                   help="Comma-separated component columns for sensitivity. Example: Exposure,Sensitivity,AdaptiveCapacity")
    p.add_argument("--weights_json", default=None,
                   help='JSON dict of weights. Example: \'{"Exposure":0.33,"Sensitivity":0.33,"AdaptiveCapacity":0.34}\'')
    p.add_argument("--weights_file", default=None, help="Optional JSON file with weights dict.")
    p.add_argument("--mc_draws", type=int, default=3000, help="Monte Carlo draws for weight sensitivity.")
    p.add_argument("--seed", type=int, default=7, help="Random seed.")
    return p.parse_args()


def main():
    args = parse_args()
    outdir = ensure_outdir(args.outdir)

    df = pd.read_csv(args.input)
    if args.date_col not in df.columns:
        raise KeyError(f"date_col='{args.date_col}' not found in columns: {df.columns.tolist()}")
    if args.vi_col not in df.columns:
        raise KeyError(f"vi_col='{args.vi_col}' not found in columns: {df.columns.tolist()}")

    df[args.date_col] = pd.to_datetime(df[args.date_col], errors="coerce")
    if df[args.date_col].isna().mean() > 0.05:
        raise ValueError("Too many invalid dates. Check date parsing or date column.")
    df = df.sort_values(args.date_col).reset_index(drop=True)

    df[args.vi_col] = safe_float_series(df[args.vi_col], args.vi_col)

    # Optional resample
    if args.freq:
        df = (
            df.set_index(args.date_col)
            .resample(args.freq)
            .mean(numeric_only=True)
            .reset_index()
        )
        df = df.dropna(subset=[args.vi_col]).reset_index(drop=True)

    y = df[args.vi_col].to_numpy()

    # ---- Structural breaks (level)
    br_main = detect_breaks(y, df[args.date_col], method="pelt_rbf", pen=args.break_pen, min_size=args.min_size)
    br_alt = detect_breaks(y, df[args.date_col], method="binseg_l2", pen=args.break_pen, min_size=args.min_size)

    # Pettitt single break check
    cp, pval = pettitt_test(y)
    br_pett = BreakResult(indices=[cp], dates=[df[args.date_col].iloc[cp]], method=f"pettitt(p≈{pval:.3f})")

    # ---- Variance breaks
    y_centered = y - np.nanmean(y)
    v = y_centered**2
    br_var = detect_breaks(v, df[args.date_col], method="pelt_rbf", pen=args.break_pen, min_size=args.min_size)

    # Save break tables
    break_table = pd.DataFrame(
        [
            {"method": br_main.method, "break_index": i, "break_date": d}
            for i, d in zip(br_main.indices, br_main.dates)
        ]
        + [
            {"method": br_alt.method, "break_index": i, "break_date": d}
            for i, d in zip(br_alt.indices, br_alt.dates)
        ]
        + [
            {"method": br_pett.method, "break_index": i, "break_date": d}
            for i, d in zip(br_pett.indices, br_pett.dates)
        ]
        + [
            {"method": f"variance_{br_var.method}", "break_index": i, "break_date": d}
            for i, d in zip(br_var.indices, br_var.dates)
        ]
    )
    break_table.to_csv(os.path.join(outdir, "breakpoints_table.csv"), index=False)

    # Regime summaries using main breaks
    regime_summary = summarize_regimes(df, br_main.indices, args.date_col, args.vi_col)
    regime_summary.to_csv(os.path.join(outdir, "regime_summary.csv"), index=False)

    # Plots
    plot_breaks(df, args.date_col, args.vi_col, br_main, os.path.join(outdir, "VI_breaks_pelt.png"))
    plot_breaks(df, args.date_col, args.vi_col, br_pett, os.path.join(outdir, "VI_breaks_pettitt.png"))

    # Variance break plot (simple)
    plt.figure()
    plt.plot(df[args.date_col], v, label="(VI - mean)^2")
    for d in br_var.dates:
        plt.axvline(d, linestyle="--")
    plt.title(f"Variance break detection ({br_var.method})")
    plt.xlabel("Date")
    plt.ylabel("Squared deviation")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "VI_variance_breaks.png"), dpi=200)
    plt.close()

    # ---- Sensitivity analysis
    sensitivity_outputs = []

    if args.components:
        components = [c.strip() for c in args.components.split(",") if c.strip()]
        missing = [c for c in components if c not in df.columns]
        if missing:
            raise KeyError(f"Missing component columns: {missing}. Present columns: {df.columns.tolist()}")

        # Load weights
        weights = None
        if args.weights_json:
            weights = json.loads(args.weights_json)
        elif args.weights_file:
            with open(args.weights_file, "r", encoding="utf-8") as f:
                weights = json.load(f)
        else:
            raise ValueError("For sensitivity, provide weights via --weights_json or --weights_file")

        # Validate weights cover all components
        for c in components:
            if c not in weights:
                raise KeyError(f"Weight for component '{c}' is missing.")
        weights = normalize_weights({c: float(weights[c]) for c in components})

        thresholds = [0.70, 0.75, 0.80]

        # 1) Monte Carlo weight sensitivity
        for pct in [0.10, 0.20, 0.30]:
            mc = monte_carlo_weight_sensitivity(
                df=df,
                components=components,
                weights=weights,
                pct=pct,
                draws=args.mc_draws,
                thresholds=thresholds,
                seed=args.seed,
            )
            sensitivity_outputs.append(mc)

        # 2) Leave-one-out
        loo = leave_one_out_sensitivity(df, components, weights)
        loo.to_csv(os.path.join(outdir, "sensitivity_leave_one_out.csv"), index=False)

        # 3) Normalization sensitivity
        base = compute_vi_from_components(df, components, weights)
        for nm in ["minmax", "zscore", "robust"]:
            alt = normalization_sensitivity(df, components, weights, norm_method=nm)
            row = pd.DataFrame(
                [{
                    "scenario": f"normalization_{nm}",
                    "metric": "spearman",
                    "p05": spearman(base, alt),
                    "p50": spearman(base, alt),
                    "p95": spearman(base, alt),
                },
                {
                    "scenario": f"normalization_{nm}",
                    "metric": "kendall",
                    "p05": kendall(base, alt),
                    "p50": kendall(base, alt),
                    "p95": kendall(base, alt),
                },
                {
                    "scenario": f"normalization_{nm}",
                    "metric": "rmse",
                    "p05": float(np.sqrt(np.mean((alt-base)**2))),
                    "p50": float(np.sqrt(np.mean((alt-base)**2))),
                    "p95": float(np.sqrt(np.mean((alt-base)**2))),
                }]
            )
            # add threshold stability
            for thr in thresholds:
                stab = classification_stability(base, alt, thr)
                row = pd.concat(
                    [row,
                     pd.DataFrame([{
                         "scenario": f"normalization_{nm}",
                         "metric": f"class_stability_thr_{thr:.2f}",
                         "p05": stab, "p50": stab, "p95": stab
                     }])]
                )
            sensitivity_outputs.append(row)

        # Save MC summaries
        sens_summary = pd.concat(sensitivity_outputs, ignore_index=True)
        sens_summary.to_csv(os.path.join(outdir, "sensitivity_summary.csv"), index=False)

        # Optional plot: baseline vs alt normalization
        plt.figure()
        plt.plot(df[args.date_col], base, label="baseline_components_VI")
        plt.plot(df[args.date_col], normalization_sensitivity(df, components, weights, "minmax"), label="minmax")
        plt.plot(df[args.date_col], normalization_sensitivity(df, components, weights, "zscore"), label="zscore")
        plt.plot(df[args.date_col], normalization_sensitivity(df, components, weights, "robust"), label="robust")
        plt.title("VI sensitivity to normalization")
        plt.xlabel("Date")
        plt.ylabel("VI (reconstructed)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "VI_normalization_sensitivity.png"), dpi=200)
        plt.close()

    else:
        # Still save threshold sensitivity on existing VI
        thresholds = [0.70, 0.75, 0.80]
        rows = []
        for thr in thresholds:
            rows.append({"threshold": thr, "share_high_vulnerability": float((y >= thr).mean())})
        pd.DataFrame(rows).to_csv(os.path.join(outdir, "threshold_sensitivity_on_VI.csv"), index=False)

    # Done
    print("\n=== COMPLETED ===")
    print(f"Outputs saved to: {outdir}")
    print("Key files:")
    print("  - breakpoints_table.csv")
    print("  - regime_summary.csv")
    print("  - VI_breaks_pelt.png")
    print("  - VI_variance_breaks.png")
    if args.components:
        print("  - sensitivity_summary.csv")
        print("  - sensitivity_leave_one_out.csv")
        print("  - VI_normalization_sensitivity.png")
    else:
        print("  - threshold_sensitivity_on_VI.csv")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[ERROR]", str(e))
        print("\nTip: Run with correct columns. Example:\n"
              "python structural_breaks_sensitivity_vi.py --input vi.csv "
              "--date_col date --vi_col VI --outdir results_vi "
              "--components Exposure,Sensitivity,AdaptiveCapacity "
              "--weights_json '{\"Exposure\":0.33,\"Sensitivity\":0.33,\"AdaptiveCapacity\":0.34}'\n")
        raise