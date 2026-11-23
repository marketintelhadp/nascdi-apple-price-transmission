import sys; sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "..", ".."))
from src.revision import rev_lib as L
from src.revision.rev_run import estimate_chain
nas = L.load_nascdi()
rows = []
for k in (0, 6, 10):
    print("flag runs of >=", k, "identical weekly values as missing")
    for name, path in L.chains():
        try:
            res, _ = estimate_chain(name, path, nas["NASCDI"], label=f"s{k}", stale=k, cv_reps=1500, boot=0)
            rows.append(dict(chain=name, threshold=k, n=res['n'], ECT=res['ECT'], F=res['F'], F_U5=res['F_U5'], t=res['t'], t_U5=res['t_U5'], decF=res['bounds_F_decision5'], dect=res['bounds_t_decision5']))
            print(f"  {name:20s} n={res['n']:3d} ECT {res['ECT']:.3f} F {res['F']:.2f} (U5 {res['F_U5']:.2f}) t {res['t']:.2f} (U5 {res['t_U5']:.2f}) {res['bounds_F_decision5']}/{res['bounds_t_decision5']}", flush=True)
        except Exception as e:
            print(f"  {name:20s} not estimable ({type(e).__name__})")

import os; pd = __import__('pandas'); pd.DataFrame(rows).to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'results_revision', 'audit', 'stale_price_filter.csv'), index=False)
