"""Print the full results table (baselines, round 1, round 2, nested selection, level-only references) as
markdown, for README.md.

    uv run --project /Users/aryan/Projects/hacktheburgh/keyboard-fatigue-nobaseline/analysis python make_table.py
"""

from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
COLS = ["lopo:MAE", "lopo:skill", "lopo:within_concordance", "rolling:MAE", "rolling:skill",
        "rolling:within_concordance"]
HEAD = ["LOPO MAE", "LOPO skill", "LOPO conc", "rolling MAE", "rolling skill", "rolling conc"]


def main():
    r1 = pd.read_csv(HERE / "results_round1.csv", index_col=0)
    r2 = pd.read_csv(HERE / "results_round2.csv", index_col=0)
    z = pd.read_csv(HERE / "level_only_refs.csv", index_col=0)
    z = z.loc[[i for i in z.index if i.startswith("Z")]]
    allr = pd.concat([r1[COLS], r2[COLS], z[COLS]]).astype(float)
    v1 = allr.loc["v1"]
    lines = ["| variant | " + " | ".join(HEAD) + " |", "|---|" + "---:|" * len(HEAD)]
    for name, row in allr.iterrows():
        cells = []
        for c in COLS:
            val = row[c]
            better = (val < v1[c] - 1e-9) if c.endswith("MAE") else (val > v1[c] + 1e-9)
            txt = f"{val:.3f}"
            cells.append(f"**{txt}**" if better and name not in ("v1", "guess", "clock", "v3") else txt)
        lines.append(f"| {name.replace('|', '¦')} | " + " | ".join(cells) + " |")
    print("\n".join(lines))
    deploy = allr.drop(index=["guess", "clock", "v1", "v3"])
    both = deploy[(deploy["lopo:MAE"] < v1["lopo:MAE"]) & (deploy["rolling:MAE"] < v1["rolling:MAE"])]
    conc = deploy[(deploy["lopo:within_concordance"] > v1["lopo:within_concordance"])
                  & (deploy["rolling:within_concordance"] >= v1["rolling:within_concordance"])]
    print(f"\n{len(deploy)} rows besides the baselines; beat v1 on MAE in both schemes: {len(both)}; "
          f"beat v1 on LOPO concordance without losing rolling concordance: {len(conc)} {list(conc.index)}")

    perm = HERE / "perm_results.json"
    if perm.exists():
        import json
        res = json.loads(perm.read_text())
        keys = [c.replace(":", ":") for c in COLS]
        print("\nObserved value (permutation p):\n")
        print("| model | " + " | ".join(HEAD) + " |")
        print("|---|" + "---:|" * len(HEAD))
        for name, rec in res.items():
            cells = [f"{rec['observed'][k.split(':')[0]][k.split(':')[1]]:.3f} ({rec['p'][k]:.3f})" for k in keys]
            print(f"| {name} | " + " | ".join(cells) + " |")
        print("\nGain over v1 on the real labels, mean gain over v1 on shuffled labels, and the share of shuffles "
              "with a gain at least as large (p). MAE gains are negative when better:\n")
        print("| model | " + " | ".join(HEAD) + " |")
        print("|---|" + "---:|" * len(HEAD))
        for name, rec in res.items():
            if name == "v1":
                continue
            cells = [f"{rec['gain_vs_v1'][k]['observed_gain']:+.3f} / {rec['gain_vs_v1'][k]['null_mean_gain']:+.3f} "
                     f"(p {rec['gain_vs_v1'][k]['p']:.2f})" for k in keys]
            print(f"| {name} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
