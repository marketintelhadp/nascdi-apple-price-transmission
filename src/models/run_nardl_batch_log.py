import argparse
import glob
import hashlib
import json
import os
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from statsmodels.tools.tools import add_constant


IN_DIR = "data/model"
OUT_DIR = "results/nardl"
MAX_LAGS_Y = 4
MAX_LAGS_X = 4
MIN_OBS_AFTER_LAGS = 60
HAC_MAXLAGS = 4


def file_sha256(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def csv_row_count(path: Path) -> int:
    if not path.exists() or not path.is_file():
        return 0
    return max(sum(1 for _ in path.open("r", encoding="utf-8", errors="ignore")) - 1, 0)


def parse_name(fname: str):
    """
    Parse weekly_model_<producer>_<variety>_<grade>_posneg.csv.
    Producer can contain underscores.
    """
    base = os.path.basename(fname).replace("_posneg.csv", "")
    m = re.match(r"weekly_model_(.+)_(American|Delicious|Maharaji)_(A|B)$", base)
    if not m:
        return ("Unknown", "Unknown", "Unknown")
    return m.group(1), m.group(2), m.group(3)


def price_columns(dependent: str):
    if dependent == "producer":
        return "log_price_prod", "log_price_term", "producer", "terminal"
    return "log_price_term", "log_price_prod", "terminal", "producer"


def build_ardl_design(df, max_lags_y: int, max_lags_x: int, dependent: str):
    """
    ECM/NARDL design:
    dy_t = c + a*y_{t-1} + b*x_{t-1} + c1*p_{t-1} + c2*n_{t-1}
           + lagged dy/dx/dp/dn terms + e_t.
    """
    d = df.sort_values("week_end").copy()
    y, x, y_label, x_label = price_columns(dependent)
    p = "NASCDI_pos"
    n = "NASCDI_neg"

    d["dy"] = d[y].diff()
    d["dx"] = d[x].diff()
    d["dp"] = d[p].diff()
    d["dn"] = d[n].diff()

    d["y_L1"] = d[y].shift(1)
    d["x_L1"] = d[x].shift(1)
    d["p_L1"] = d[p].shift(1)
    d["n_L1"] = d[n].shift(1)

    for lag in range(1, max_lags_y + 1):
        d[f"dy_L{lag}"] = d["dy"].shift(lag)
    for lag in range(0, max_lags_x + 1):
        d[f"dx_L{lag}"] = d["dx"].shift(lag)
        d[f"dp_L{lag}"] = d["dp"].shift(lag)
        d[f"dn_L{lag}"] = d["dn"].shift(lag)

    cols_needed = (
        ["dy", "y_L1", "x_L1", "p_L1", "n_L1"]
        + [f"dy_L{lag}" for lag in range(1, max_lags_y + 1)]
        + [f"dx_L{lag}" for lag in range(0, max_lags_x + 1)]
        + [f"dp_L{lag}" for lag in range(0, max_lags_x + 1)]
        + [f"dn_L{lag}" for lag in range(0, max_lags_x + 1)]
    )

    d2 = d.dropna(subset=cols_needed).copy()
    y_vec = d2["dy"]
    x_cols = (
        ["y_L1", "x_L1", "p_L1", "n_L1"]
        + [f"dy_L{lag}" for lag in range(1, max_lags_y + 1)]
        + [f"dx_L{lag}" for lag in range(0, max_lags_x + 1)]
        + [f"dp_L{lag}" for lag in range(0, max_lags_x + 1)]
        + [f"dn_L{lag}" for lag in range(0, max_lags_x + 1)]
    )
    x_mat = add_constant(d2[x_cols], has_constant="add")
    return d2, y_vec, x_mat, y_label, x_label


def bounds_like_summary(model):
    names = model.params.index.tolist()
    r_mat = np.zeros((4, len(names)))
    for i, name in enumerate(["y_L1", "x_L1", "p_L1", "n_L1"]):
        if name in names:
            r_mat[i, names.index(name)] = 1.0
    ftest = model.f_test(r_mat)
    return float(np.asarray(ftest.fvalue)), float(np.asarray(ftest.pvalue))


def long_run_effects(model):
    a = model.params.get("y_L1", np.nan)
    b = model.params.get("x_L1", np.nan)
    c = model.params.get("p_L1", np.nan)
    d = model.params.get("n_L1", np.nan)
    if np.isnan(a) or a == 0:
        return np.nan, np.nan, np.nan
    return -b / a, -c / a, -d / a


def write_metadata(args, files, res_out, diag_out, skip_out, rows_count, skipped_count):
    out_dir = Path(args.out_dir)
    outputs = [Path(res_out), Path(diag_out), Path(skip_out)]
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": "src/models/run_nardl_batch_log.py",
        "python": sys.version,
        "platform": platform.platform(),
        "arguments": vars(args),
        "model_specification": {
            "dependent": args.dependent,
            "max_lags_y": args.max_lags_y,
            "max_lags_x": args.max_lags_x,
            "min_obs_after_lags": args.min_obs_after_lags,
            "hac_maxlags": args.hac_maxlags,
            "covariance": "HAC",
        },
        "inputs": {
            Path(fp).name: {
                "path": str(fp),
                "sha256": file_sha256(Path(fp)),
                "rows": csv_row_count(Path(fp)),
            }
            for fp in files
        },
        "outputs": {
            p.name: {
                "path": str(p),
                "sha256": file_sha256(p),
                "rows": csv_row_count(p) if p.suffix.lower() == ".csv" else None,
            }
            for p in outputs
            if p.exists()
        },
        "run_summary": {
            "estimated_series": int(rows_count),
            "skipped_series": int(skipped_count),
        },
    }
    meta_path = out_dir / "nardl_run_metadata.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    return meta_path


def main():
    parser = argparse.ArgumentParser(
        description="Run log-price NARDL/ECM batch models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--in_dir", default=IN_DIR, help="Directory containing weekly_model_*_posneg.csv files.")
    parser.add_argument("--out_dir", default=OUT_DIR, help="Directory for NARDL outputs.")
    parser.add_argument(
        "--dependent",
        choices=["terminal", "producer"],
        default="terminal",
        help="Price side used as the dependent variable. Use producer if the manuscript models farm/producer prices.",
    )
    parser.add_argument("--max_lags_y", type=int, default=MAX_LAGS_Y)
    parser.add_argument("--max_lags_x", type=int, default=MAX_LAGS_X)
    parser.add_argument("--min_obs_after_lags", type=int, default=MIN_OBS_AFTER_LAGS)
    parser.add_argument("--hac_maxlags", type=int, default=HAC_MAXLAGS)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.in_dir, "weekly_model_*_posneg.csv")))
    if not files:
        raise FileNotFoundError(f"No model files found in {args.in_dir}")

    rows = []
    diag = []
    skipped_rows = []

    for fp in files:
        market, variety, grade = parse_name(fp)
        df = pd.read_csv(fp, parse_dates=["week_end"])

        required = [
            "avg_price_prod",
            "avg_price_term",
            "log_price_prod",
            "log_price_term",
            "NASCDI",
            "NASCDI_pos",
            "NASCDI_neg",
        ]
        missing = [col for col in required if col not in df.columns]
        if missing:
            skipped_rows.append(
                {
                    "file": fp,
                    "producer_market": market,
                    "variety": variety,
                    "grade": grade,
                    "reason": f"missing columns: {missing}",
                }
            )
            continue

        df = df.dropna(subset=required).sort_values("week_end")
        if len(df) < args.min_obs_after_lags:
            skipped_rows.append(
                {
                    "file": fp,
                    "producer_market": market,
                    "variety": variety,
                    "grade": grade,
                    "reason": f"too short pre-lags: {len(df)}",
                }
            )
            continue

        d2, y_vec, x_mat, y_label, x_label = build_ardl_design(
            df,
            max_lags_y=args.max_lags_y,
            max_lags_x=args.max_lags_x,
            dependent=args.dependent,
        )
        if len(d2) < args.min_obs_after_lags:
            skipped_rows.append(
                {
                    "file": fp,
                    "producer_market": market,
                    "variety": variety,
                    "grade": grade,
                    "reason": f"too short after lags: {len(d2)}",
                }
            )
            continue

        model = OLS(y_vec, x_mat).fit(cov_type="HAC", cov_kwds={"maxlags": args.hac_maxlags})
        fval, fpval = bounds_like_summary(model)
        lr_x, lr_pos, lr_neg = long_run_effects(model)

        resid = model.resid
        lb = acorr_ljungbox(resid, lags=[8], return_df=True)
        lb_p = float(lb["lb_pvalue"].iloc[0])
        bp = het_breuschpagan(resid, model.model.exog)
        bp_p = float(bp[1])

        rows.append(
            {
                "producer_market": market,
                "variety": variety,
                "grade": grade,
                "dependent_price": y_label,
                "regressor_price": x_label,
                "n_obs": len(d2),
                "ECT_y_L1": model.params.get("y_L1", np.nan),
                "ECT_pvalue": model.pvalues.get("y_L1", np.nan),
                "LR_pass_through_regressor": lr_x,
                "LR_pass_through_prod": lr_x if x_label == "producer" else np.nan,
                "LR_pass_through_term": lr_x if x_label == "terminal" else np.nan,
                "LR_effect_NASCDI_pos": lr_pos,
                "LR_effect_NASCDI_neg": lr_neg,
                "bounds_F": fval,
                "bounds_F_pvalue": fpval,
                "R2": model.rsquared,
                "AIC": model.aic,
                "BIC": model.bic,
            }
        )

        diag.append(
            {
                "producer_market": market,
                "variety": variety,
                "grade": grade,
                "dependent_price": y_label,
                "regressor_price": x_label,
                "ljungbox_p_8": lb_p,
                "bp_heterosk_p": bp_p,
            }
        )

    if not rows:
        skip_df = pd.DataFrame(skipped_rows)
        skip_out = out_dir / "nardl_skipped_log.csv"
        skip_df.to_csv(skip_out, index=False)
        raise RuntimeError(
            "No series estimated. Inspect nardl_skipped_log.csv and rebuild model inputs if needed."
        )

    res_df = pd.DataFrame(rows).sort_values(["producer_market", "variety", "grade"])
    diag_df = pd.DataFrame(diag).sort_values(["producer_market", "variety", "grade"])
    skip_df = pd.DataFrame(skipped_rows)

    res_out = out_dir / "nardl_results_log.csv"
    diag_out = out_dir / "nardl_diagnostics_log.csv"
    skip_out = out_dir / "nardl_skipped_log.csv"

    res_df.to_csv(res_out, index=False)
    diag_df.to_csv(diag_out, index=False)
    skip_df.to_csv(skip_out, index=False)

    meta_path = write_metadata(
        args=args,
        files=files,
        res_out=res_out,
        diag_out=diag_out,
        skip_out=skip_out,
        rows_count=len(res_df),
        skipped_count=len(skip_df),
    )

    print(f"Estimated: {len(res_df)} | Skipped: {len(skip_df)}")
    print(f"Saved: {res_out}")
    print(f"Saved: {diag_out}")
    print(f"Saved: {skip_out}")
    print(f"Saved metadata: {meta_path}")


if __name__ == "__main__":
    main()
