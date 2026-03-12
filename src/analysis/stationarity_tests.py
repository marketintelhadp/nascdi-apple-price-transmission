import os
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

DATA_DIR = "data/model"
OUTPUT_FILE = "results/stationarity_tests.csv"

def adf_test(series):
    result = adfuller(series.dropna(), autolag='AIC')
    return result[0], result[1]

def kpss_test(series):
    stat, p, lags, crit = kpss(series.dropna(), regression='c', nlags="auto")
    return stat, p

def run_tests():

    rows = []

    for file in os.listdir(DATA_DIR):

        if not file.endswith(".csv"):
            continue

        df = pd.read_csv(os.path.join(DATA_DIR, file))

        variables = [
            "log_price_prod",
            "log_price_term",
            "NASCDI",
            "NASCDI_pos",
            "NASCDI_neg"
        ]

        for var in variables:

            if var not in df.columns:
                continue

            adf_stat, adf_p = adf_test(df[var])
            kpss_stat, kpss_p = kpss_test(df[var])

            rows.append({
                "dataset": file,
                "variable": var,
                "ADF_stat": adf_stat,
                "ADF_pvalue": adf_p,
                "KPSS_stat": kpss_stat,
                "KPSS_pvalue": kpss_p
            })

    results = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    results.to_csv(OUTPUT_FILE, index=False)

    print("Stationarity tests saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    run_tests()