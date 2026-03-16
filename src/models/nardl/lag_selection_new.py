import os
import pandas as pd
from statsmodels.tsa.ardl import ardl_select_order

DATA_DIR = "data/model"
OUTPUT_FILE = "results/lag_selection_table.csv"

rows = []

for file in os.listdir(DATA_DIR):

    if not file.endswith(".csv"):
        continue

    path = os.path.join(DATA_DIR, file)
    df = pd.read_csv(path)

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

    # Extract lag structure
    ar_lag = max(model.ar_lags) if model.ar_lags else 0

    pos_lag = None
    neg_lag = None

    if hasattr(model, "dl_lags"):
        pos_lag = max(model.dl_lags.get("NASCDI_pos", [0]))
        neg_lag = max(model.dl_lags.get("NASCDI_neg", [0]))

    rows.append({
        "dataset": file,
        "AR_lag": ar_lag,
        "NASCDI_pos_lag": pos_lag,
        "NASCDI_neg_lag": neg_lag,
        "AIC": sel.aic
    })

results = pd.DataFrame(rows)

os.makedirs("results", exist_ok=True)
results.to_csv(OUTPUT_FILE, index=False)

print("\nLag selection table saved to:")
print(OUTPUT_FILE)
print(results)