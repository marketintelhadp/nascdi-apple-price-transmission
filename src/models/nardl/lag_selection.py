import os
import pandas as pd
from statsmodels.tsa.ardl import ardl_select_order

DATA_DIR = "data/model"
OUTPUT_DIR = "results"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "lag_selection_results.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

rows = []

for file in os.listdir(DATA_DIR):

    if not file.endswith(".csv"):
        continue

    path = os.path.join(DATA_DIR, file)

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"Could not read {file}: {e}")
        continue

    required_cols = ["log_price_prod", "NASCDI_pos", "NASCDI_neg"]

    if not all(c in df.columns for c in required_cols):
        print(f"Skipping {file} (missing required columns)")
        continue

    df = df.dropna()

    y = df["log_price_prod"]
    X = df[["NASCDI_pos", "NASCDI_neg"]]

    try:

        sel = ardl_select_order(
            endog=y,
            maxlag=6,
            exog=X,
            maxorder=6,
            ic="aic",
            trend="c"
        )

        model = sel.model
        fitted = model.fit()

        # Extract lag structure
        lag_y = max(model.ar_lags) if model.ar_lags else 0

        lag_pos = None
        lag_neg = None

        if hasattr(model, "_order"):
            order = model._order
            lag_pos = order.get("NASCDI_pos", 0)
            lag_neg = order.get("NASCDI_neg", 0)

        rows.append({
            "dataset": file,
            "lag_y": lag_y,
            "lag_NASCDI_pos": lag_pos,
            "lag_NASCDI_neg": lag_neg,
            "AIC": round(fitted.aic,4),
            "BIC": round(fitted.bic,4),
            "HQIC": round(fitted.hqic,4)
        })

        print(f"Saved results for {file}")

    except Exception as e:
        print(f"Error processing {file}: {e}")

results = pd.DataFrame(rows)

print("\nFinal Lag Selection Table:\n")
print(results)

results.to_csv(OUTPUT_FILE, index=False)

try:
    results.to_excel(os.path.join(OUTPUT_DIR, "lag_selection_results.xlsx"), index=False)
except:
    pass

print("\nResults saved to:", OUTPUT_FILE)