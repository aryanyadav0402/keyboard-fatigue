import sys, math
sys.path.insert(0, "/Users/aryan/Projects/hacktheburgh/keyboard-fatigue-nobaseline/analysis/results/experiments")
import numpy as np, pandas as pd
import harness as h
from load_data import pair_presses, MODIFIER_KEYS

T = h.load_table(); D = h.scored_panel(T)
RAW = h.raw_sessions()

def seq(cid):
    """log IKI (down-to-down of non-modifier presses, task 0) and log hold, aligned per press; backspace flag."""
    ev = [e for e in RAW[cid]["keys"] if e[0] == 0]
    pr = [p for p in pair_presses(ev) if p["key"] not in MODIFIER_KEYS]
    dn = np.array([p["down"] for p in pr]); up = np.array([p["up"] for p in pr], float)
    bs = np.array([p["key"] == "Backspace" for p in pr], float)
    iki = np.diff(dn)
    hold = (up - dn)[1:]
    ok = (iki > 0) & (iki < 5000) & np.isfinite(hold) & (hold > 0) & (hold < 2000)
    return np.log(iki[ok]), np.log(hold[ok]), bs[1:][ok]

SEQ = {c: seq(c) for c in T.checkin_id if c in RAW}
BASE = {p: [c for c in g.checkin_id if c in SEQ] for p, g in T[T.is_baseline].groupby("pid")}
if __name__ == "__main__":
    print(len(SEQ), {p: len(v) for p, v in BASE.items()})
    print(pd.Series({c: len(SEQ[c][0]) for c in D.checkin_id}).describe())
