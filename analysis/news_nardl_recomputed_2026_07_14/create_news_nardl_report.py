"""Create reproducible tables and figures for the targeted-GDELT news NASCDI NARDL run."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import jarque_bera
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.diagnostic import (
    acorr_breusch_godfrey,
    acorr_ljungbox,
    breaks_cusumolsresid,
    het_arch,
    het_breuschpagan,
)
from statsmodels.tools.tools import add_constant


MAX_LAGS_Y = 4
MAX_LAGS_X = 4
HAC_MAXLAGS = 4
HORIZON = 20


def parse_name(path: Path) -> tuple[str, str, str]:
    match = re.match(r"weekly_model_(.+)_(American|Delicious|Maharaji)_(A|B)_posneg", path.stem)
    if not match:
        raise ValueError(f"Unexpected model filename: {path.name}")
    return match.group(1).replace("_", " "), match.group(2), match.group(3)


def design(df: pd.DataFrame):
    d = df.sort_values("week_end").copy()
    y, x, p, n = "log_price_term", "log_price_prod", "NASCDI_pos", "NASCDI_neg"
    d["dy"] = d[y].diff()
    d["dx"] = d[x].diff()
    d["dp"] = d[p].diff()
    d["dn"] = d[n].diff()
    for name, col in (("y_L1", y), ("x_L1", x), ("p_L1", p), ("n_L1", n)):
        d[name] = d[col].shift(1)
    for lag in range(1, MAX_LAGS_Y + 1):
        d[f"dy_L{lag}"] = d["dy"].shift(lag)
    for lag in range(0, MAX_LAGS_X + 1):
        for prefix, col in (("dx", "dx"), ("dp", "dp"), ("dn", "dn")):
            d[f"{prefix}_L{lag}"] = d[col].shift(lag)
    cols = (
        ["y_L1", "x_L1", "p_L1", "n_L1"]
        + [f"dy_L{i}" for i in range(1, MAX_LAGS_Y + 1)]
        + [f"{prefix}_L{i}" for prefix in ("dx", "dp", "dn") for i in range(0, MAX_LAGS_X + 1)]
    )
    d = d.dropna(subset=["dy"] + cols).copy()
    return d, d["dy"], add_constant(d[cols], has_constant="add")


def wald_equal(model, left: list[str], right: list[str]) -> tuple[float, float]:
    names = list(model.params.index)
    r = np.zeros((1, len(names)))
    for name in left:
        r[0, names.index(name)] += 1
    for name in right:
        r[0, names.index(name)] -= 1
    test = model.wald_test(r, scalar=True)
    return float(test.statistic), float(test.pvalue)


def simulate_response(params: pd.Series, shock: str) -> np.ndarray:
    """Level response to a one-unit permanent partial-sum innovation at t=0."""
    y = np.zeros(HORIZON + 2)
    dy = np.zeros(HORIZON + 2)
    p = np.zeros(HORIZON + 2)
    n = np.zeros(HORIZON + 2)
    if shock == "positive":
        p[1:] = 1.0
    else:
        n[1:] = 1.0
    for t in range(1, HORIZON + 2):
        value = params.get("y_L1", 0.0) * y[t - 1]
        value += params.get("p_L1", 0.0) * p[t - 1]
        value += params.get("n_L1", 0.0) * n[t - 1]
        for lag in range(1, MAX_LAGS_Y + 1):
            if t - lag >= 0:
                value += params.get(f"dy_L{lag}", 0.0) * dy[t - lag]
        for lag in range(0, MAX_LAGS_X + 1):
            if t - lag >= 0:
                value += params.get(f"dp_L{lag}", 0.0) * (p[t - lag] - p[t - lag - 1] if t - lag - 1 >= 0 else p[t - lag])
                value += params.get(f"dn_L{lag}", 0.0) * (n[t - lag] - n[t - lag - 1] if t - lag - 1 >= 0 else n[t - lag])
        dy[t] = value
        y[t] = y[t - 1] + dy[t]
    return y[1:]


def save_plot(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--nascdi_path", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir)
    table_dir, plot_dir = out_dir / "tables", out_dir / "plots"
    table_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    summary_rows, param_rows, diag_rows, dynamic_rows, price_rows = [], [], [], [], []
    for path in sorted(model_dir.glob("weekly_model_*_posneg.csv")):
        market, variety, grade = parse_name(path)
        frame = pd.read_csv(path, parse_dates=["week_end"])
        d, response, regressors = design(frame)
        model = OLS(response, regressors).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_MAXLAGS})
        params = model.params
        a = params["y_L1"]
        lr_price, lr_pos, lr_neg = -params["x_L1"] / a, -params["p_L1"] / a, -params["n_L1"] / a
        lr_stat, lr_p = wald_equal(model, ["p_L1"], ["n_L1"])
        sr_stat, sr_p = wald_equal(model, [f"dp_L{i}" for i in range(MAX_LAGS_X + 1)], [f"dn_L{i}" for i in range(MAX_LAGS_X + 1)])
        residuals = model.resid
        lb = acorr_ljungbox(residuals, lags=[8], return_df=True).iloc[0]
        bg = acorr_breusch_godfrey(OLS(response, regressors).fit(), nlags=8)
        bp = het_breuschpagan(residuals, regressors)
        arch = het_arch(residuals, nlags=8)
        jb = jarque_bera(residuals)
        cusum_stat, cusum_p, _ = breaks_cusumolsresid(residuals, ddof=int(model.df_model))
        label = f"{market} | {variety} | Grade {grade}"
        summary_rows.append({
            "producer_market": market, "variety": variety, "grade": grade, "label": label,
            "n_obs": len(d), "ECT_y_L1": a, "ECT_pvalue_HAC": model.pvalues["y_L1"],
            "LR_pass_through_producer": lr_price, "LR_effect_NASCDI_pos": lr_pos,
            "LR_effect_NASCDI_neg": lr_neg, "LR_asymmetry_Wald": lr_stat,
            "LR_asymmetry_pvalue_HAC": lr_p, "SR_asymmetry_Wald": sr_stat,
            "SR_asymmetry_pvalue_HAC": sr_p, "bounds_like_F": float(model.f_test(np.eye(4, len(params), 1)).fvalue),
            "R2": model.rsquared, "AIC": model.aic, "BIC": model.bic,
        })
        diag_rows.append({
            "producer_market": market, "variety": variety, "grade": grade, "n_obs": len(d),
            "LjungBox_p_lag8": float(lb["lb_pvalue"]), "BreuschGodfrey_p_lag8": float(bg[1]),
            "BreuschPagan_p": float(bp[1]), "ARCH_p_lag8": float(arch[1]), "JarqueBera_p": float(jb.pvalue),
            "CUSUM_stat": float(cusum_stat), "CUSUM_pvalue": float(cusum_p),
        })
        for name in params.index:
            param_rows.append({"producer_market": market, "variety": variety, "grade": grade,
                               "parameter": name, "coefficient": params[name], "HAC_se": model.bse[name],
                               "HAC_t": model.tvalues[name], "HAC_pvalue": model.pvalues[name]})
        for shock in ("positive", "negative"):
            values = simulate_response(params, shock)
            dynamic_rows.extend({"producer_market": market, "variety": variety, "grade": grade,
                                 "shock": shock, "horizon_week": i, "log_price_response": value,
                                 "percent_response_approx": 100 * value} for i, value in enumerate(values))
        price_rows.append(frame.assign(label=label))

    summary = pd.DataFrame(summary_rows).sort_values(["producer_market", "variety", "grade"])
    diagnostics = pd.DataFrame(diag_rows).sort_values(["producer_market", "variety", "grade"])
    pd.DataFrame(param_rows).sort_values(["producer_market", "variety", "grade", "parameter"]).to_csv(table_dir / "nardl_hac_parameters.csv", index=False)
    summary.to_csv(table_dir / "nardl_news_summary.csv", index=False)
    diagnostics.to_csv(table_dir / "nardl_news_diagnostics.csv", index=False)
    dynamics = pd.DataFrame(dynamic_rows)
    dynamics.to_csv(table_dir / "nardl_dynamic_responses.csv", index=False)

    nascdi = pd.read_csv(args.nascdi_path, parse_dates=["week_end"]).sort_values("week_end")
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(nascdi["week_end"], nascdi["NASCDI"], color="#17324d", linewidth=1.0, label="Weekly news NASCDI")
    ax.axhline(100, color="#9b2226", linestyle="--", linewidth=1, label="Normalization mean")
    ax.set(title="Targeted GDELT news NASCDI", xlabel="Week", ylabel="Index (mean = 100)")
    ax.legend(frameon=False, ncol=2)
    save_plot(fig, plot_dir / "fig_01_weekly_news_nascdi.png")

    fig, axes = plt.subplots(4, 2, figsize=(14, 12), sharex=True)
    for ax, frame in zip(axes.flat, price_rows):
        ax.plot(frame["week_end"], frame["avg_price_prod"], color="#2a9d8f", label="Producer")
        ax.plot(frame["week_end"], frame["avg_price_term"], color="#e76f51", label="Terminal")
        ax2 = ax.twinx()
        ax2.plot(frame["week_end"], frame["NASCDI"], color="#264653", alpha=.35, linewidth=.8, label="NASCDI")
        ax.set_title(frame["label"].iloc[0], fontsize=9)
        ax.set_ylabel("Price")
        ax2.set_ylabel("NASCDI", color="#264653")
    axes[0, 0].legend(loc="upper left", fontsize=8, frameon=False)
    fig.suptitle("Weekly price series and targeted-GDELT news NASCDI", y=1.01)
    save_plot(fig, plot_dir / "fig_02_price_nascdi_context.png")

    display = summary.copy()
    display["label"] = display["producer_market"] + " | " + display["variety"] + " | G" + display["grade"]
    display = display.sort_values("LR_effect_NASCDI_pos")
    y = np.arange(len(display))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(display["LR_effect_NASCDI_pos"], y, color="#d1495b", s=48, label="NASCDI intensification (+)")
    ax.scatter(display["LR_effect_NASCDI_neg"], y, color="#00798c", s=48, marker="s", label="NASCDI easing (-)")
    ax.axvline(0, color="black", linewidth=.8)
    ax.set_yticks(y, display["label"], fontsize=8)
    ax.set(xlabel="Long-run effect on log terminal price per NASCDI unit", title="Asymmetric long-run news-NASCDI effects")
    ax.legend(frameon=False)
    save_plot(fig, plot_dir / "fig_03_long_run_asymmetry.png")

    fig, ax = plt.subplots(figsize=(10, 6))
    speed = summary.sort_values("ECT_y_L1")
    yy = np.arange(len(speed))
    colors = np.where(speed["ECT_pvalue_HAC"] < .05, "#2a9d8f", "#b0b0b0")
    ax.hlines(yy, 0, speed["ECT_y_L1"], color=colors, linewidth=2)
    ax.scatter(speed["ECT_y_L1"], yy, color=colors, s=38)
    ax.axvline(0, color="black", linewidth=.8)
    ax.set_yticks(yy, speed["producer_market"] + " | " + speed["variety"] + " | G" + speed["grade"], fontsize=8)
    ax.set(xlabel="Error-correction coefficient", title="Speed of adjustment (green: HAC p < 0.05)")
    save_plot(fig, plot_dir / "fig_04_error_correction_speed.png")

    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharex=True)
    for ax, (_, group) in zip(axes.flat, dynamics.groupby(["producer_market", "variety", "grade"], sort=True)):
        title = f"{group['producer_market'].iloc[0]} | {group['variety'].iloc[0]} | G{group['grade'].iloc[0]}"
        for shock, color, label in (("positive", "#d1495b", "Intensification (+)"), ("negative", "#00798c", "Easing (-)")):
            one = group[group["shock"] == shock]
            ax.plot(one["horizon_week"], one["percent_response_approx"], color=color, label=label)
        ax.axhline(0, color="black", linewidth=.6)
        ax.set_title(title, fontsize=9)
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.supxlabel("Weeks after a one-unit NASCDI partial-sum shock")
    fig.supylabel("Approx. terminal-price response (%)")
    fig.suptitle("Dynamic NARDL responses to news-NASCDI shocks", y=1.01)
    save_plot(fig, plot_dir / "fig_05_dynamic_nardl_responses.png")

    print(f"Saved tables to: {table_dir}")
    print(f"Saved figures to: {plot_dir}")


if __name__ == "__main__":
    main()
