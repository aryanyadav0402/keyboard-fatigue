"""Keystroke-level features for the rule sentence (task 0), from a replay of the text buffer.

Editing keys in this data (checked on all 36 check-ins): character keys, Shift, Backspace (including
32 auto-repeat key-downs from holding it) and one Enter per check-in (submission). The app's input
stream has only insertText and deleteContentBackward and the caret always sits at the end, so the
buffer is a stack: a character key pushes, a Backspace key-down (repeat or not) pops.

Errors: after every key we keep d = min over j of Levenshtein(buffer, expected[:j]), the distance
from the buffer to the best-matching start of the expected answer, with one DP row per buffer
character (a stack, so Backspace restores the row exactly). Pushing a character raises d by 0 or 1;
a push that raises it is an error keystroke. Because rows are restored on Backspace, d always equals
the number of error keystrokes still in the buffer, so "errors left" is the final d.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # the analysis folder
from features import rule_characters, typing_features  # noqa: E402
from load_data import CAPS_LOCK_FLAG, MODIFIER_KEYS, REPEAT_FLAG  # noqa: E402
from tasks import apply_rule, score  # noqa: E402

TAIL_RELATIVE = 2.0  # a "tail" gap is longer than twice the check-in's own median ordinary gap


# ---- replay ----------------------------------------------------------------------------------

def rule_unit(session: dict) -> dict:
    units = [u for u in session["units"] if u["task"] == 0 and u.get("kind") == "transform"]
    assert len(units) == 1, "expected exactly one rule sentence per check-in"
    u = dict(units[0])
    computed = apply_rule(u.get("question_type"), u["source"])
    u["expected"] = computed if computed is not None else u["expected"]  # recompute, as load_data does
    return u


def midpoint(expected: str) -> int:
    """Index where the second half starts: just after the word separator nearest the middle."""
    seps = [i for i, c in enumerate(expected) if c in " -."]
    best = min(seps, key=lambda i: abs(i + 1 - len(expected) / 2)) if seps else len(expected) // 2
    return best + 1


def word_index(expected: str) -> list[int]:
    """Word number of every position; a separator belongs to the word before it."""
    out, w = [], 0
    for c in expected:
        out.append(w)
        if c in " -.":
            w += 1
    return out


def part_chars(meta: dict, split: str, part: int | None) -> int:
    """How many expected characters a part of the sentence has."""
    m = len(meta["expected"])
    if part is None:
        return m
    if split == "half":
        return meta["mid"] if part == 1 else m - meta["mid"]
    return sum(1 for w in word_index(meta["expected"]) if (w % 2 == 0) == (part == 1))


def classify_error(typed: str, target: str | None, src: str | None) -> str:
    """'rule' when the slip is about the rule (forgot to apply it, applied it where it doesn't belong,
    or used the wrong separator); 'typo' for any other wrong character."""
    if target is None:
        return "typo"  # extra characters past the end
    if src is not None and target != src and typed == src:
        return "rule"  # e.g. 'a' for 'A', ' ' for '-'
    if src is not None and target == src and typed != target and typed.lower() == target.lower():
        return "rule"  # a capital where none was asked for
    if target in " -." and typed in " -.":
        return "rule"  # wrong separator, e.g. ' ' or '-' where '.' was asked for
    return "typo"


def replay(session: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """One row per key-down of the rule sentence with the buffer state, plus one row per press
    (key-down paired with key-up, auto-repeats skipped, as load_data.pair_presses does)."""
    u = rule_unit(session)
    expected, source = u["expected"], u["source"]
    m = len(expected)
    mid = midpoint(expected)
    fields = session.get("key_fields")
    rows = [dict(zip(fields, r)) for r in session["keys"]]
    rows = [r for r in rows if r["task"] == 0 and r.get("unit", 0) == 0]
    order = sorted(range(len(rows)), key=lambda i: (rows[i]["t"], i))
    rows = [rows[i] for i in order]

    buf, dp, slot_event = [], [list(range(m + 1))], []
    events, presses, held = [], [], {}
    run_id, in_run = -1, False
    other_keys = []
    for r in rows:
        flags = int(r.get("flags") or 0)
        key, t = r["key"], float(r["t"])
        if r["type"] == "u":
            i = held.pop(r["code"], None)
            if i is not None:
                presses[i]["up_ms"] = t
            continue
        repeat = bool(flags & REPEAT_FLAG)
        pos = len(buf)
        if not repeat:
            presses.append({"key": key, "code": r["code"], "down_ms": t, "up_ms": math.nan,
                            "caps_lock": bool(flags & CAPS_LOCK_FLAG), "pos": pos})
            held[r["code"]] = len(presses) - 1
        ev = {"t": t, "key": key, "repeat": repeat, "pos": pos, "d_before": min(dp[-1])}
        if key == "Backspace":
            ev["kind"] = "bs"
            if not in_run:
                run_id += 1
                in_run = True
            ev["run"] = run_id
            if buf:
                buf.pop()
                dp.pop()
                ev["popped"] = slot_event.pop()
            else:
                ev["popped"] = -1
        elif len(key) == 1:
            ev["kind"] = "ins"
            in_run = False
            prev = dp[-1]
            j_star = min(range(m + 1), key=lambda j: (prev[j], abs(j - pos)))
            row = [prev[0] + 1]
            for j in range(1, m + 1):
                row.append(min(prev[j] + 1, row[j - 1] + 1, prev[j - 1] + (key != expected[j - 1])))
            dp.append(row)
            buf.append(key)
            slot_event.append(len(events))
            ev["err"] = min(row) > ev["d_before"]
            ev["target"] = expected[j_star] if j_star < m else None
            ev["src"] = source[j_star] if j_star < m and len(source) == m else None
            ev["etype"] = classify_error(key, ev["target"], ev["src"]) if ev["err"] else None
        elif key in MODIFIER_KEYS:
            ev["kind"] = "mod"
        elif key == "Enter":
            ev["kind"] = "enter"
        else:
            ev["kind"] = "other"
            other_keys.append(key)
        ev["d_after"] = min(dp[-1])
        events.append(ev)

    ev = pd.DataFrame(events)
    word_of = word_index(expected)
    for col in ("popped", "run", "err", "target", "src", "etype"):
        if col not in ev:
            ev[col] = np.nan
    ev["deleted"] = False
    ev["deleted_in_run"] = np.nan
    for i, e in ev[ev.kind.eq("bs") & ev.popped.ge(0)].iterrows():
        ev.loc[int(e.popped), "deleted"] = True
        ev.loc[int(e.popped), "deleted_in_run"] = e.run
    # which part of the sentence each key belongs to, by the position it acts on (an insert at pos,
    # a deletion of the character at pos - 1): first/second half, or odd/even word
    ev["apos"] = np.where(ev.kind.eq("bs"), np.maximum(ev.pos - 1, 0), ev.pos)
    ev["half"] = np.where(ev.apos < mid, 1, 2)
    ev["oddeven"] = np.where(ev.apos.map(lambda i: word_of[min(i, m - 1)]) % 2 == 0, 1, 2)
    pr = pd.DataFrame(presses)
    pr["apos"] = np.where(pr.key.eq("Backspace"), np.maximum(pr.pos - 1, 0), pr.pos)
    pr["half"] = np.where(pr.apos < mid, 1, 2)
    pr["oddeven"] = np.where(pr.apos.map(lambda i: word_of[min(i, m - 1)]) % 2 == 0, 1, 2)
    final = "".join(buf)
    typed = (u.get("typed") or "").rstrip("\n")
    meta = {"expected": expected, "source": source, "typed": typed, "final": final, "mid": mid,
            "replay_matches_typed": final == typed, "d_final": int(min(dp[-1])),
            "other_keys": other_keys, "question_type": u.get("question_type"), "t_shown": u["t_shown"]}
    return ev, pr, meta


# ---- gaps ------------------------------------------------------------------------------------

def clean_gaps(ev: pd.DataFrame, meta: dict, split: str = "half") -> pd.DataFrame:
    """Gaps between consecutive character key-downs with no Backspace between them and neither key an
    error. 'ordinary' when both characters are ones the rule leaves alone (lowercase letters, and
    spaces when the rule keeps them), as in features.py."""
    changed, kept = rule_characters(meta["source"], meta["expected"])
    keyed = ev[ev.kind.isin(["ins", "bs"])].reset_index()
    out = []
    for a, b in zip(keyed.iloc[:-1].itertuples(), keyed.iloc[1:].itertuples()):
        if a.kind == "ins" and b.kind == "ins" and not a.err and not b.err:
            pa, pb = getattr(a, split), getattr(b, split)
            out.append({"gap": b.t - a.t, "ordinary": a.key in kept and b.key in kept,
                        "part": pb if pa == pb else 0, "idx": b.index})
    return pd.DataFrame(out, columns=["gap", "ordinary", "part", "idx"])


# ---- features ----------------------------------------------------------------------------------

def nanmed(x):
    x = [v for v in x if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(np.median(x)) if x else math.nan


def keystroke_features(ev: pd.DataFrame, meta: dict, half: int | None = None,
                       ref_log_gaps: np.ndarray | None = None, split: str = "half") -> dict:
    """Error, correction and timing features for the whole sentence or one half of it.

    ref_log_gaps: log ordinary gaps from the person's baseline check-ins, for baseline-relative
    features (none for walk-ups)."""
    n_chars = part_chars(meta, split, half)
    sel = ev if half is None else ev[ev[split].eq(half)]
    ins, bs = sel[sel.kind.eq("ins")], sel[sel.kind.eq("bs")]
    errs = ins[ins.err.astype(bool)]
    out = {"n_ins": len(ins), "n_del": int((bs.popped >= 0).sum()),
           "err_per100": 100 * len(errs) / n_chars,
           "err_rule_per100": 100 * int(errs.etype.eq("rule").sum()) / n_chars,
           "err_typo_per100": 100 * int(errs.etype.eq("typo").sum()) / n_chars}
    fixed = errs[errs.deleted]
    out["fix_rate"] = len(fixed) / len(errs) if len(errs) else math.nan
    out["left_per100"] = 100 * (len(errs) - len(fixed)) / n_chars
    # deletions of characters that were not errors: collateral (in a run that also removes an error)
    # or spurious (in a run that removes no error at all)
    popped = bs[bs.popped >= 0]
    err_runs = set(ev.loc[ev.kind.eq("ins") & ev.err.astype(bool) & ev.deleted, "deleted_in_run"].dropna())
    popped_err = popped.popped.map(lambda i: bool(ev.loc[int(i), "err"]))
    spurious = popped[~popped_err.astype(bool) & ~popped.run.isin(err_runs)]
    out["spurious_del_per100"] = 100 * len(spurious) / n_chars

    # detection: inserts between an error and the start of the run that deletes it, and the time
    keyed = ev[ev.kind.isin(["ins", "bs"])]
    run_start = ev[ev.kind.eq("bs")].groupby("run").t.min()
    run_end = ev[ev.kind.eq("bs")].groupby("run").t.max()
    delay_keys, delay_ms = [], []
    for i, e in fixed.iterrows():
        start = run_start[e.deleted_in_run]
        between = keyed[(keyed.t > e.t) & (keyed.t < start)]
        delay_keys.append(int(between.kind.eq("ins").sum()))
        delay_ms.append(start - e.t)
    out["detect_keys"] = float(np.mean(delay_keys)) if delay_keys else math.nan
    out["detect_ms"] = nanmed(delay_ms)
    out["immediate_share"] = float(np.mean(np.asarray(delay_keys) == 0)) if delay_keys else math.nan

    # pace references, from the whole sentence even for a half (keeps ratios comparable)
    gaps_all = clean_gaps(ev, meta)
    ordinary_all = gaps_all[gaps_all.ordinary].gap.to_numpy(float)
    pace_ord = float(np.median(ordinary_all)) if len(ordinary_all) >= 5 else math.nan
    pace_all = float(np.median(gaps_all.gap)) if len(gaps_all) >= 5 else math.nan

    # post-error slowing: the gap right after an error key (before any correction), the gap from the
    # end of a correction run to the next character, and the next two gaps after that
    keyed_list = keyed.reset_index()
    pos_of = {idx: k for k, idx in enumerate(keyed_list["index"])}
    post_err = []
    for i, e in errs.iterrows():
        k = pos_of[i]
        if k + 1 < len(keyed_list):
            post_err.append((keyed_list.t.iloc[k + 1] - e.t) / pace_all)
    out["post_err_ratio"] = nanmed(post_err)
    resume, pes = [], []
    runs_here = sel[sel.kind.eq("bs")].run.dropna().unique()
    for r in runs_here:
        end = run_end[r]
        after = keyed_list[keyed_list.t > end]
        if after.empty or after.kind.iloc[0] != "ins":
            continue
        resume.append((after.t.iloc[0] - end) / pace_all)
        nxt = after.iloc[:3]
        if len(nxt) == 3 and (nxt.kind == "ins").all():
            pes.append(float(np.mean(np.diff(nxt.t.to_numpy(float)))) / pace_all)
    out["resume_ratio"] = nanmed(resume)
    out["pes_ratio"] = nanmed(pes)

    # timing distribution of clean ordinary gaps
    g = gaps_all if half is None else clean_gaps(ev, meta, split).query("part == @half")
    og = g[g.ordinary].gap.to_numpy(float)
    og = og[og > 0]
    if len(og) >= 5:
        lg = np.log(og)
        q10, q50, q90 = np.quantile(og, [0.1, 0.5, 0.9])
        out.update(iki_q10=float(q10), iki_q50=float(q50), iki_q90=float(q90),
                   iki_logsd=float(np.std(lg, ddof=1)),
                   tail_share=float(np.mean(og > TAIL_RELATIVE * pace_ord)))
        if ref_log_gaps is not None and len(ref_log_gaps) >= 5:
            ref90 = float(np.quantile(ref_log_gaps, 0.9))
            out["tail_base90"] = float(np.mean(lg > ref90))
            out["w1_base"] = float(stats.wasserstein_distance(lg, ref_log_gaps))
            out["ks_base"] = float(stats.ks_2samp(lg, ref_log_gaps).statistic)
    return out


def log_ordinary_gaps(ev: pd.DataFrame, meta: dict) -> np.ndarray:
    g = clean_gaps(ev, meta)
    og = g[g.ordinary].gap.to_numpy(float)
    return np.log(og[og > 0])


# ---- the v1 features on half a sentence, for split-half reliability -------------------------------

def v1_half_features(pr: pd.DataFrame, meta: dict, half: int | None) -> dict:
    """features.typing_features on one half's presses (or all of them), so the v1 features can be
    measured on each half of the sentence. Accuracy compares the matching halves of the typed and
    expected text."""
    mid, typed, expected, source = meta["mid"], meta["typed"], meta["expected"], meta["source"]
    if half is None:
        p, ty, ex, so = pr, typed, expected, source
    else:
        p = pr[pr.half.eq(half)]
        sl = slice(0, mid) if half == 1 else slice(mid, None)
        ty, ex, so = typed[sl], expected[sl], source[sl]
    f = typing_features(p[["key", "down_ms", "up_ms", "caps_lock"]].reset_index(drop=True), ty, ex, so, rule=True)
    keep = ["rule_cost", "backspace_per100", "accuracy", "gap_cv", "pause1s_share", "hold_med", "gap_med",
            "wpm", "pause_rel_share"]
    return {k: f.get(k, math.nan) for k in keep}


def v1_native(pr: pd.DataFrame, meta: dict, split: str = "half", part: int | None = None) -> dict:
    """The v1 Task 1 features computed directly (same definitions as features.typing_features) on any
    part of the sentence, counting only gaps whose two keys both fall in that part. Accuracy compares
    the part's positions of the typed and expected text (NaN if the lengths differ)."""
    from features import char_class, iqr
    expected, typed, source = meta["expected"], meta["typed"], meta["source"]
    changed, kept = rule_characters(source, expected)
    chars = pr[~pr.key.isin(MODIFIER_KEYS) & (pr.key != "Enter")].sort_values("down_ms").reset_index(drop=True)
    inside = np.ones(len(chars), bool) if part is None else chars[split].eq(part).to_numpy()
    out = {}
    n_keys = int(inside.sum())
    if n_keys < 5:
        return out
    out["backspace_per100"] = 100 * int((chars.key.eq("Backspace").to_numpy() & inside).sum()) / n_keys
    keys, downs, ups = chars.key.tolist(), chars.down_ms.to_numpy(float), chars.up_ms.to_numpy(float)
    classes = [char_class(k, changed, kept) for k in keys]
    ordinary, touching, every = [], [], []
    for i in range(1, len(keys)):
        if not (inside[i] and inside[i - 1]) or "other" in (classes[i - 1], classes[i]):
            continue
        gap = downs[i] - downs[i - 1]
        every.append(gap)
        (touching if "rule" in (classes[i - 1], classes[i]) else ordinary).append(gap)
    if len(ordinary) >= 5:
        out["gap_cv"] = iqr(ordinary) / float(np.median(ordinary))
        out["gap_med"] = float(np.median(ordinary))
    if len(every) >= 5:
        out["pause1s_share"] = float(np.mean(np.asarray(every) > 1000))
    if len(touching) >= 3 and len(ordinary) >= 3:
        out["rule_cost"] = float(np.median(touching)) / float(np.median(ordinary))
    letters = chars[inside & chars.key.str.len().eq(1).to_numpy() & chars.key.str.isalpha().to_numpy()]
    holds = (letters.up_ms - letters.down_ms).dropna()
    holds = holds[(holds > 0) & (holds < 2000)]
    if len(holds) >= 5:
        out["hold_med"] = float(holds.median())
    if part is None:
        out["accuracy"] = score(typed, expected)["accuracy"]
    elif len(typed) == len(expected):
        if split == "half":
            sl = slice(0, meta["mid"]) if part == 1 else slice(meta["mid"], None)
            out["accuracy"] = score(typed[sl], expected[sl])["accuracy"]
        else:
            idx = [i for i, w in enumerate(word_index(expected)) if (w % 2 == 0) == (part == 1)]
            out["accuracy"] = score("".join(typed[i] for i in idx), "".join(expected[i] for i in idx))["accuracy"]
    return out
