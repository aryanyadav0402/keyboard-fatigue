import json
from common import h
rows = []
for f, notes in (("exp1_results.json", "kernel/distance"), ("exp2_results.json", "windows 16/4"), ("exp3_results.json", "window variants / v1+seq")):
    for n, r in json.load(open(f)).items():
        h.log_result("sequence", n, 28, r, notes); rows.append((n, r))
with open("table.md", "w") as fh:
    fh.write("| model | lopo MAE | lopo skill | lopo conc | rolling MAE | rolling skill | rolling conc |\n|---|---|---|---|---|---|---|\n")
    for n, r in rows:
        l, o = r["lopo"], r["rolling"]
        fh.write(f"| {n} | {l['MAE']:.3f} | {l['skill']:+.3f} | {l['within_concordance']:.3f} | {o['MAE']:.3f} | {o['skill']:+.3f} | {o['within_concordance']:.3f} |\n")
