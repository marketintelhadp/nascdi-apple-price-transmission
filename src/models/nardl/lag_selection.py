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

        # Extract dependent lag
        lag_y = max(model.ar_lags) if model.ar_lags else 0

        # Extract NASCDI lags
        lag_pos = None
        lag_neg = None

        if hasattr(model, "dl_lags") and model.dl_lags:
            lag_pos = max(model.dl_lags.get("NASCDI_pos", [0]))
            lag_neg = max(model.dl_lags.get("NASCDI_neg", [0]))

        # Extract best IC values safely
        best_aic_value = float(sel.aic.loc[sel.aic.idxmin()])
        best_bic_value = float(sel.bic.loc[sel.bic.idxmin()])
        best_hqic_value = float(sel.hqic.loc[sel.hqic.idxmin()])

        print(
            f"{file} | lag_y={lag_y}, lag_pos={lag_pos}, lag_neg={lag_neg}, "
            f"AIC={best_aic_value:.3f}"
        )

        rows.append({
            "dataset": file,
            "lag_y": lag_y,
            "lag_NASCDI_pos": lag_pos,
            "lag_NASCDI_neg": lag_neg,
            "AIC": round(best_aic_value,4),
            "BIC": round(best_bic_value,4),
            "HQIC": round(best_hqic_value,4)
        })

    except Exception as e:

        print(f"Error processing {file}: {e}")
        continue


# Convert to DataFrame
results = pd.DataFrame(rows)

print("\nFinal lag selection table:")
print(results)

# Save results
results.to_csv(OUTPUT_FILE, index=False)

try:
    results.to_excel(os.path.join(OUTPUT_DIR, "lag_selection_results.xlsx"), index=False)
except:
    print("Excel export skipped (openpyxl not installed).")

print("\nLag selection results saved to:")
print(OUTPUT_FILE)