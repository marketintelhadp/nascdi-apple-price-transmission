"""Main revised estimation: calendar-week lags, selected lags, simulated bounds, symmetry tests."""
import sys
import pandas as pd

from . import rev_lib as L
from .rev_run import estimate_chain

if __name__ == "__main__":
    label = sys.argv[1] if len(sys.argv) > 1 else "base"
    nas = L.load_nascdi()
    rows = []
    for name, path in L.chains():
        res, _ = estimate_chain(name, path, nas["NASCDI"], label=label)
        rows.append(res)
        print(name, "lags", (res["p"], res["q"], res["r"]), "n", res["n"], "ECT", round(res["ECT"], 3),
              "F", round(res["F"], 2), "dec", res["bounds_F_decision5"], res["bounds_t_decision5"],
              "LRp", round(res["LR_p_boot"], 3), "SRp", round(res["SR_p_boot"], 3), "final", res["final_spec"], flush=True)
    pd.DataFrame(rows).to_csv(L.OUT_DIR / f"main_{label}.csv", index=False)
    print("saved")
