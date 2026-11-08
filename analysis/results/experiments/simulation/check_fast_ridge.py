"""Check that simlib's fast LOPO/rolling matches the harness numbers on the real scored table."""
import sys
sys.path.insert(0, "/Users/aryan/Projects/hacktheburgh/keyboard-fatigue-nobaseline/analysis/results/experiments")
sys.path.insert(0, "..")
import numpy as np
from types import SimpleNamespace
import harness as H
import simlib as S
t = H.load_table(); d = H.scored_panel(t)
pid = d.pid.astype("category").cat.codes.to_numpy()
tab = SimpleNamespace(pid=pid, kss=d.kss.to_numpy(float), base=d.kss_base.to_numpy(float), clk=d.hours_rel.to_numpy(float),
    lp=np.log(d.checkin_n.to_numpy(float)), t=d.timestamp.astype("int64").to_numpy() / 3.6e12,
    X=d[H.TYPING + ["h_rate_dev", "h_errr_dev"]].to_numpy(float))
for name, fp in S.BASELINES.items():
    m1 = {s: S.metrics(tab, *p(tab, fp)) for s, p in (("lopo", S.lopo_predict), ("rolling", S.rolling_predict))}
    h = H.evaluate(d, H.BASELINES[name])
    print(name, "fast lopo MAE/skill %.3f %.3f | harness %.3f %.3f" % (m1["lopo"]["MAE"], m1["lopo"]["skill"], h["lopo"]["MAE"], h["lopo"]["skill"]),
          "| rolling fast %.3f %.3f harness %.3f %.3f" % (m1["rolling"]["MAE"], m1["rolling"]["skill"], h["rolling"]["MAE"], h["rolling"]["skill"]))
