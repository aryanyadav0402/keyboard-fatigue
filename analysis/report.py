"""Write the report, figures, predictions and features from the model results."""

import math
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from load_data import Tables  # noqa: E402
from model import MAIN  # noqa: E402

TIME = mdates.DateFormatter("%H:%M")


def num(x, digits: int = 2) -> str:
    if x is None or (isinstance(x, (float, np.floating)) and not np.isfinite(x)):
        return "–"
    return f"{x:.{digits}f}"


def pval(p) -> str:
    if p is None or not np.isfinite(p):
        return "–"
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def p_eq(p) -> str:
    text = pval(p)
    return f"p {text}" if text.startswith("<") else f"p = {text}"


def time_axis(ax) -> None:
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 4)))
    ax.xaxis.set_major_formatter(TIME)


def predictions_table(f: pd.DataFrame, r: dict) -> pd.DataFrame:
    """Out-of-fold predictions for scored panel check-ins, and no-baseline ones for one-offs."""
    out = f[["checkin_id", "person", "timestamp"]].copy()
    out["tiredness_pred"] = np.nan
    out["tired_prob"] = np.nan
    if "preds" in r:
        ids = r["scored"].checkin_id
        out["tiredness_pred"] = out.checkin_id.map(dict(zip(ids, r["preds"][MAIN])))
        out["tired_prob"] = out.checkin_id.map(dict(zip(ids, r["tired_prob"])))
    if r.get("external"):
        pred = r["external"]["pred"]
        out.loc[pred.index, "tiredness_pred"] = pred.to_numpy()
    return out


def save(fig, path: Path) -> str:
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path.name


def make_figures(f: pd.DataFrame, r: dict, folder: Path) -> dict:
    folder.mkdir(parents=True, exist_ok=True)
    made = {}
    panel = f[f.participant_type.eq("panel")].sort_values("timestamp") if len(f) else f
    people = sorted(panel.person.unique()) if len(panel) else []
    if not people:
        return made

    pred = dict(zip(r["scored"].checkin_id, r["preds"][MAIN])) if "preds" in r else {}
    cols = min(4, len(people))
    fig, axes = plt.subplots(math.ceil(len(people) / cols), cols, figsize=(3.3 * cols, 2.6 * math.ceil(len(people) / cols)),
                             squeeze=False, sharey=True)
    for ax, person in zip(axes.flat, people):
        g = panel[panel.person == person]
        ax.plot(g.timestamp, g.kss, "o", color="#333333", ms=4, label="rated")
        g = g[g.checkin_id.isin(pred.keys())]
        if len(g):
            ax.plot(g.timestamp, g.checkin_id.map(pred), "-", color="#d1495b", label="predicted")
        ax.set_title(person, fontsize=9)
        ax.set_ylim(0.5, 9.5)
        time_axis(ax)
        ax.tick_params(labelsize=7)
    for ax in axes.flat[len(people):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=7)
    fig.suptitle("Rated vs predicted tiredness (KSS 1-9), each person predicted by a model trained on the others",
                 fontsize=10)
    made["tiredness"] = save(fig, folder / "tiredness_over_night.png")

    if "vt_rule_cost" in panel and panel.vt_rule_cost.notna().any():
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for person, g in panel.dropna(subset=["hours_awake", "vt_rule_cost"]).groupby("person"):
            ax.plot(g.hours_awake, g.vt_rule_cost, "-o", ms=3, lw=1, label=person)
        ax.set_xlabel("hours awake")
        ax.set_ylabel("rule cost (gap next to a capital ÷ ordinary gap)")
        ax.set_title("Vowel-rule cost against hours awake, in time order")
        ax.legend(fontsize=7, ncol=2)
        made["rule_cost"] = save(fig, folder / "rule_cost_vs_hours.png")

    if "beyond_clock" in r:
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
        b, o = r["beyond_clock"], r["original"]
        axes[0].hist(b["null"], bins=30, color="#9aa5b1")
        axes[0].axvline(b["improvement"], color="#d1495b", lw=2, label=f"real data, {p_eq(b['p'])}")
        axes[0].set_xlabel("error cut by adding typing to hours awake (KSS points)")
        axes[0].set_title("Main test: typing shuffled within each person", fontsize=10)
        axes[1].hist(o["null"], bins=30, color="#9aa5b1")
        axes[1].axvline(o["mae"], color="#d1495b", lw=2, label=f"real data, {p_eq(o['p'])}")
        axes[1].set_xlabel("typing-only error (KSS points); lower is better")
        axes[1].set_title("Original test: ratings shuffled within each person", fontsize=10)
        for ax in axes:
            ax.legend(fontsize=8)
        made["permutation"] = save(fig, folder / "permutation_tests.png")

    if "rt_median" in panel and panel.rt_median.notna().any():
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for person, g in panel.dropna(subset=["rt_median"]).groupby("person"):
            ax.plot(g.timestamp, g.rt_median, "-o", ms=3, lw=1, label=person)
        time_axis(ax)
        ax.set_ylabel("median reaction time (ms)")
        ax.set_title("Reaction time over the night")
        ax.legend(fontsize=7, ncol=2)
        made["rt"] = save(fig, folder / "reaction_time_over_night.png")

    s = r.get("sleep")
    if s is not None and len(s):
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
        for _, row in s.iterrows():
            for ax, col in zip(axes, ["kss", "rule_cost", "rt_median"]):
                ax.plot([0, 1], [row[f"{col}_before"], row[f"{col}_after"]], "-o", label=row.person)
        for ax, label in zip(axes, ["KSS", "rule cost", "median reaction time (ms)"]):
            ax.set_xticks([0, 1], ["before sleep", "after sleep"])
            ax.set_ylabel(label)
        axes[0].legend(fontsize=7)
        made["sleep"] = save(fig, folder / "before_after_sleep.png")
    return made


def mixed_section(name: str, m: dict) -> list[str]:
    lines = [f"### {name}", "", f"`{m['formula']}`, random intercept per person.", ""]
    if "error" in m:
        return lines + [f"Skipped: {m['error']}.", ""]
    note = "" if m["converged"] else " The fit did not fully converge, so treat it with caution."
    lines += [f"{m['n']} check-ins from {m['people']} people.{note}", "",
              "| Term | Coefficient | 95% CI | p |", "|---|---|---|---|"]
    for term, row in m["table"].iterrows():
        lines.append(f"| {term} | {num(row.coef, 3)} | {num(row.ci_low, 3)} to {num(row.ci_high, 3)} | {pval(row.p)} |")
    return lines + [""]


def findings(r: dict) -> list[str]:
    out = []
    if "beyond_clock" in r:
        b = r["beyond_clock"]
        if b["p"] < 0.05 and b["improvement"] > 0:
            out.append(f"Typing adds to hours awake: adding the typing features cut the average error from "
                       f"{num(b['clock_mae'])} to {num(b['main_mae'])} KSS points (shuffle test {p_eq(b['p'])}).")
        else:
            out.append(f"Typing did not clearly add to hours awake: the average error went from {num(b['clock_mae'])} "
                       f"to {num(b['main_mae'])} KSS points (shuffle test {p_eq(b['p'])}). That's a null result.")
        o = r["original"]
        text = f"The original test, which shuffles ratings within each person, gives {p_eq(o['p'])}."
        if o["p"] < 0.05 and b["p"] >= 0.05:
            text += (" It would call typing predictive, but only because typing and tiredness both drift over the"
                     " night; the main test shows typing adds nothing beyond the clock.")
        out.append(text)
    rule = r.get("mixed", {}).get("Rule cost", {})
    if "table" in rule and "hours_awake" in rule["table"].index:
        row = rule["table"].loc["hours_awake"]
        verdict = ("so tired brains do pay more for the rule" if row.ci_low > 0
                   else "so the rule cost fell rather than rose" if row.ci_high < 0
                   else "which isn't clearly different from zero")
        out.append(f"After allowing for practice, the rule cost changed by {num(row.coef, 3)} per hour awake "
                   f"(95% CI {num(row.ci_low, 3)} to {num(row.ci_high, 3)}), {verdict}.")
    c = r.get("rule_control")
    if c:
        out.append(f"Gaps next to a capital took {num(c['median_ratio'])}× as long as ordinary gaps "
                   f"(Wilcoxon {p_eq(c['p'])}, {c['n']} check-ins), so the vowel rule does cost time.")
    s = r.get("sleep")
    if s is not None and len(s):
        out.append(f"Across {len(s)} sleeps, KSS changed by {num((s.kss_after - s.kss_before).mean(), 1)} on average "
                   f"and the rule cost by {num((s.rule_cost_after - s.rule_cost_before).mean())}.")
    e = r.get("external")
    if e:
        better = "better than" if e["MAE"] < e["constant_MAE"] else "no better than"
        out.append(f"For {e['people']} one-off people, who have no baseline, the model was {better} always guessing "
                   f"the panel average (error {num(e['MAE'])} vs {num(e['constant_MAE'])} KSS points).")
    return out


def report_text(tables: Tables, f: pd.DataFrame, r: dict, figures: dict, data_dir: str) -> str:
    panel = f[f.participant_type.eq("panel")] if len(f) else f
    oneoff = f[f.participant_type.eq("oneoff")] if len(f) else f
    lines = ["# Tiredness from typing: results", ""]
    summary = (f"Data from `{data_dir}`, analysed {datetime.now():%a %d %b %H:%M}. {len(f)} check-ins: "
               f"{len(panel)} from {panel.person.nunique() if len(panel) else 0} panel people and "
               f"{len(oneoff)} from {oneoff.person.nunique() if len(oneoff) else 0} one-off people. "
               f"{len(r['scored'])} panel check-ins are scored; each person's first 2 are their baseline.")
    if tables.skipped_test_runs:
        summary += f" {tables.skipped_test_runs} test runs were skipped."
    lines += [summary, ""]
    for task, label in (("vt", "vowel task"), ("ft", "free text")):
        col = f"{task}_focus_losses"
        if col in f and (f[col] > 0).any():
            lines.append(f"- {int((f[col] > 0).sum())} {label} timings dropped because the page lost focus.")
    if "aborted" in f and f.aborted.any():
        lines.append(f"- {int(f.aborted.sum())} check-ins were stopped early; their finished tasks are used.")
    lines += [f"- {p}" for p in tables.problems] + [""]

    if r["missing"]:
        lines += ["## Not enough data for the models yet", ""] + [f"- {m}" for m in r["missing"]] + [""]
    items = findings(r)
    if items:
        lines += ["## Findings", ""] + [f"- {x}" for x in items] + [""]

    if "table" in r:
        lines += ["## Does typing add anything to hours awake?", "",
                  "Each person is predicted by a model trained on everyone else. Anchored models predict the change "
                  "from the person's own baseline rating, then add it back.", "",
                  "| Model | Error (MAE, KSS points) | Spearman | Spearman within person (median) |",
                  "|---|---|---|---|"]
        for name, row in r["table"].iterrows():
            lines.append(f"| {name} | {num(row.MAE)} | {num(row.Spearman)} | {num(row['Spearman within person'])} |")
        b, o = r["beyond_clock"], r["original"]
        lines += ["", f"**Main test.** Adding typing to hours awake changes the error from {num(b['clock_mae'])} to "
                      f"{num(b['main_mae'])} KSS points. Shuffling the typing features within each person "
                      f"{len(b['null'])} times gives {p_eq(b['p'])}.", "",
                  f"**Original test.** Shuffling ratings within each person, as in the plan, gives a typing-only "
                  f"error of {num(o['mae'])} and {p_eq(o['p'])}. This test can't tell typing apart from anything "
                  "else that drifts over the night.", "",
                  f"**Tired vs alert** (above each person's median KSS or not): AUC {num(r['auc'])} with hours awake "
                  f"and typing, {num(r['auc_clock'])} with hours awake alone.", ""]
        if "permutation" in figures:
            lines += [f"![Permutation tests](figures/{figures['permutation']})", ""]
        if "tiredness" in figures:
            lines += [f"![Tiredness over the night](figures/{figures['tiredness']})", ""]

    if r.get("mixed"):
        lines += ["## Mixed models", ""]
        for name, m in r["mixed"].items():
            lines += mixed_section(name, m)
        if "rule_cost" in figures:
            lines += [f"![Rule cost against hours awake](figures/{figures['rule_cost']})", ""]

    c = r.get("rule_control")
    if c:
        lines += ["## Is the vowel rule really harder?", "",
                  f"Median gap next to a capital ÷ median ordinary gap: {num(c['median_ratio'])} across {c['n']} "
                  f"check-ins (Wilcoxon signed-rank {p_eq(c['p'])})."]
        if "plain_p" in c:
            lines.append(f"Vowel-task wpm ÷ plain wpm: {num(c['plain_wpm_ratio'])} (Wilcoxon {p_eq(c['plain_p'])}).")
        lines.append("")

    s = r.get("sleep")
    if s is not None and len(s):
        lines += ["## Before and after sleep", "", "| Person | Slept (min) | KSS | Rule cost | Median RT (ms) |",
                  "|---|---|---|---|---|"]
        for _, row in s.iterrows():
            lines.append(f"| {row.person} | {num(row.slept_min, 0)} | {num(row.kss_before, 0)} → {num(row.kss_after, 0)} | "
                         f"{num(row.rule_cost_before)} → {num(row.rule_cost_after)} | "
                         f"{num(row.rt_median_before, 0)} → {num(row.rt_median_after, 0)} |")
        lines.append("")
        if "sleep" in figures:
            lines += [f"![Before and after sleep](figures/{figures['sleep']})", ""]

    e = r.get("external")
    if e:
        lines += ["## One-off people", "",
                  f"A model trained on all panel check-ins, using only features that need no baseline "
                  f"({', '.join(e['features'])}), predicted {e['n']} check-ins from {e['people']} one-off people with "
                  f"an error of {num(e['MAE'])} KSS points. Always guessing the panel average gives {num(e['constant_MAE'])}.", ""]

    if "rt" in figures:
        lines += [f"![Reaction time over the night](figures/{figures['rt']})", ""]
    lines += ["## Notes", "",
              f"- Typing features in the models: {', '.join(r['typing']) or 'none yet'}.",
              f"- Fixed features with no data here: {', '.join(r['inactive']) or 'none'}.",
              f"- Reaction-test features: {', '.join(r['rt']) or 'none yet'}.",
              "- `_z` features are each person's change from their first two check-ins, divided by the baseline "
              "spread pooled across the panel. Hours awake reset after a sleep of an hour or more.",
              "- Lapses are reaction times of 355 ms or more, the cut-off for the 3-minute test.", ""]
    return "\n".join(lines)


def write_outputs(tables: Tables, f: pd.DataFrame, r: dict, out: str | Path, data_dir: str) -> Path:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    f.to_csv(out / "features.csv", index=False)
    predictions_table(f, r).to_csv(out / "predictions.csv", index=False) if len(f) else None
    figures = make_figures(f, r, out / "figures") if len(f) else {}
    (out / "report.md").write_text(report_text(tables, f, r, figures, data_dir))
    return out / "report.md"
