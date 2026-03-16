import os
import pandas as pd
from statsmodels.tsa.ardl import ardl_select_order

DATA_DIR = "data/model"
OUTPUT_FILE = "results/lag_selection_results.csv"

rows = []

for file in os.listdir(DATA_DIR):
    if not file.endswith(".csv"):
        continue

    path = os.path.join(DATA_DIR, file)
    df = pd.read_csv(path)

    required_cols = ["log_price_prod", "NASCDI_pos", "NASCDI_neg"]
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        print(f"Skipping {file} (missing columns: {missing})")
        continue

    y = df["log_price_prod"]
    X = df[["NASCDI_pos", "NASCDI_neg"]]

    sel = ardl_select_order(
        endog=y,
        maxlag=6,
        exog=X,
        maxorder=6,
        ic="aic",
        trend="c"
    )

    model = sel.model

    # Dependent variable lags
    try:
        ar_lags = model.ar_lags
    except Exception:
        ar_lags = None

    # Exogenous lags
    if hasattr(model, "dl_lags"):
        exog_lags = model.dl_lags
    elif hasattr(model, "_order"):
        exog_lags = model._order
    else:
        exog_lags = "Unavailable"

    rows.append({
        "dataset": file,
        "optimal_lag_y": ar_lags,
        "optimal_lag_exog": exog_lags,
        "aic": sel.aic,
        "bic": sel.bic,
        "hqic": sel.hqic
    })

results = pd.DataFrame(rows)

os.makedirs("results", exist_ok=True)

results.to_csv(OUTPUT_FILE, index=False)

print("Lag selection results saved to:", OUTPUT_FILE)
print(results)