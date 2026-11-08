"""Check the feature-level simulator against the project's own simulate.py (raw keystrokes -> load_data ->
features -> processed_table). simulate.py's built-in effect sizes ("moderate") at our size: 4 people, 9 check-ins
each (2 baseline + 7 scored), then v1 LOPO as in the harness. Writes simulated check-ins only under ./pipeline_sim/."""
import sys, time, warnings, contextlib, io
warnings.filterwarnings("ignore")
A = "/Users/aryan/Projects/hacktheburgh/keyboard-fatigue-nobaseline/analysis"
sys.path.insert(0, A); sys.path.insert(0, A + "/results/experiments/simulation")
import numpy as np, pandas as pd
from types import SimpleNamespace
import simulate as sim
from features import build_features, processed_table
from load_data import load_sessions
import model, simlib as S
from multiprocessing import Pool
from pathlib import Path
HERE = Path(A + "/results/experiments/simulation/pipeline_sim")

def one(seed):
    sim.PANEL, sim.ONE_OFFS, sim.CHECKINS = 4, 0, 9
    rng = np.random.default_rng(seed)
    plan = []
    for p in sim.make_people(rng):
        times, sl = sim.make_schedule(rng, p); plan.append((p, sim.make_states(rng, p, times, sl)))
    out = {}
    import json, uuid
    for world, effect in (("effect", True), ("null", False)):
        folder = HERE / f"s{seed}" / world / "sessions"; folder.mkdir(parents=True, exist_ok=True)
        wr = np.random.default_rng(seed + 1)
        for p, states in plan:
            for st in states:
                s = sim.make_session(wr, p, st, effect)
                (folder / f"{s['session_id']}.json").write_text(json.dumps(s))
        with contextlib.redirect_stdout(io.StringIO()):
            f = processed_table(build_features(load_sessions(folder)))
        d = model.scored(f)
        if len(d) < 8: continue
        tab = SimpleNamespace(pid=d.person.astype("category").cat.codes.to_numpy(), kss=d.kss.to_numpy(float), base=d.kss_base.to_numpy(float),
            clk=d.hours_rel.to_numpy(float), lp=np.log(d.checkin_n.to_numpy(float)), t=d.timestamp.astype("int64").to_numpy()/3.6e12,
            X=np.hstack([d[model.TYPING].to_numpy(float), np.zeros((len(d), 2))]))
        tab.X = np.nan_to_num(tab.X)
        m = S.metrics(tab, *S.lopo_predict(tab, S.v1)); g = S.metrics(tab, *S.lopo_predict(tab, S.clock))
        out[world] = dict(n=len(d), v1_skill=m["skill"], v1_conc=m["conc"], clock_skill=g["skill"], kss_sd=tab.kss.std(),
                          sd={c: float(d[c].std()) for c in model.TYPING})
    return out

if __name__ == "__main__":
    t = time.time()
    with Pool(9) as p: res = p.map(one, range(100, 140))
    for w in ("effect", "null"):
        r = [x[w] for x in res if w in x]
        print(w, "n_worlds", len(r), "rows", np.mean([x["n"] for x in r]), "v1 skill mean %.2f sd %.2f  P(>0) %.2f  conc %.2f | clock skill %.2f | kss sd %.2f" % (
            np.mean([x["v1_skill"] for x in r]), np.std([x["v1_skill"] for x in r]), np.mean([x["v1_skill"] > 0 for x in r]),
            np.mean([x["v1_conc"] for x in r]), np.mean([x["clock_skill"] for x in r]), np.mean([x["kss_sd"] for x in r])))
        print("   mean feature sd:", {c: round(np.mean([x["sd"][c] for x in r]), 3) for c in r[0]["sd"]})
    print("elapsed", time.time() - t)
