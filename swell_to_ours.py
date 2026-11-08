#!/usr/bin/env python3
"""Train on SWELL-KW (per-minute keystroke counts -> mental effort / stress), test on our sessions. Nothing of ours is used to fit.
SWELL features (person-centred, in SWELL within-person SD units): log(keys per minute), sqrt(error keys / keys).
Ours: the same two quantities from typing_units.csv, centred on the person's TRAIN sessions only.
  python3 swell_to_ours.py --swell swell.pkl --out out --test-from 22"""
import argparse, csv, numpy as np, pandas as pd
from scipy.stats import spearmanr
ap = argparse.ArgumentParser(); ap.add_argument('--swell', required=True); ap.add_argument('--out', default='out')
ap.add_argument('--test-from', type=float, default=22.0); ap.add_argument('--alpha', type=float, default=1.0)
ap.add_argument('--perm', type=int, default=5000); ap.add_argument('--weights-json', default=None, help='write the trained SWELL model (weights, scaling, hold-out scores) to this JSON file')
a = ap.parse_args()
d = pd.read_pickle(a.swell) if a.swell.endswith('.pkl') else pd.read_excel(a.swell, engine='odf')
t = d[d.Condition.isin(['N', 'I', 'T']) & (d.SnKeyStrokes >= 20)].copy()
g = t.groupby(['PP', 'Condition']).agg(nkeys=('SnKeyStrokes', 'sum'), err=('SnErrorKeys', 'sum'), mins=('SnKeyStrokes', 'size'),
    eff=('MentalEffort', 'first'), stress=('Stress', 'first')).reset_index().dropna(subset=['eff'])
g = g[g.groupby('PP').PP.transform('size') == 3]
g['rate'] = np.log(g.nkeys / g.mins); g['errr'] = np.sqrt(g.err / g.nkeys)
for c in ['rate', 'errr', 'eff', 'stress']: g[c + '_c'] = g[c] - g.groupby('PP')[c].transform('mean')
sdS = {c: g[c + '_c'].std() for c in ['rate', 'errr']}
Zs = np.c_[g.rate_c / sdS['rate'], g.errr_c / sdS['errr']]
def ridge(X, y, al): return np.linalg.solve(X.T @ X + al * np.eye(X.shape[1]), X.T @ y)
print(f'SWELL: {len(g)} person-condition rows, {g.PP.nunique()} people')
W = {}; LOPO = {}
for lab in ['eff', 'stress']:
    ys = (g[lab + '_c'] / g[lab + '_c'].std()).values
    pred = np.zeros(len(g))
    for p in g.PP.unique():
        m = (g.PP == p).values; pred[m] = Zs[m] @ ridge(Zs[~m], ys[~m], a.alpha)
    W[lab] = ridge(Zs, ys, a.alpha); LOPO[lab] = dict(skill=float(1 - ((pred - ys) ** 2).sum() / (ys ** 2).sum()), spearman_rho=float(spearmanr(pred, ys)[0]))
    print(f'  inside SWELL (leave-one-person-out) predicting {lab:6}: skill {1-((pred-ys)**2).sum()/(ys**2).sum():+.3f}  rho {spearmanr(pred,ys)[0]:+.2f}   weights [rate, error] {W[lab].round(3)}')

if a.weights_json:
    import json
    json.dump(dict(description='Stage-1 SWELL model: person-centred keystroke measures -> person-centred, SD-scaled MentalEffort / Stress (ridge, alpha=%g). Trained on all SWELL people; hold-out scores are leave-one-person-out.' % a.alpha,
        features=dict(rate='log(keys per minute) over the person-condition block, minutes with >=20 keys', errr='sqrt(error keys / keys)'),
        centring='subtract the person\'s own mean over their N, I and T blocks, then divide by the pooled within-person SD below',
        feature_order=['rate', 'errr'], within_person_sd=dict(rate=float(sdS['rate']), errr=float(sdS['errr'])),
        n_people=int(g.PP.nunique()), n_person_condition_rows=int(len(g)), conditions=['N', 'I', 'T'], ridge_alpha=a.alpha,
        weights=dict(effort=dict(zip(['rate', 'errr'], map(float, W['eff']))), stress=dict(zip(['rate', 'errr'], map(float, W['stress']))), intercept=0.0),
        scores_on_unseen_swell_people_leave_one_person_out={('effort' if k == 'eff' else k): v for k, v in LOPO.items()}), open(a.weights_json, 'w'), indent=2)

# ---- ours
U = pd.read_csv(f'{a.out}/typing_units.csv'); L = pd.read_csv(f'{a.out}/labels.csv'); SS = pd.read_csv(f'{a.out}/sessions.csv')
U = U[(U.kind == 'transform') & ((U.skipped == False) | (U.skipped == 0))].drop_duplicates('session_id')
U['mins'] = (U.t_end_ms - U.t_first_key_ms) / 60000
U['rate'] = np.log(U.n_presses / U.mins); U['errr'] = np.sqrt(U.n_backspace / U.n_presses)
D = U[['session_id', 'participant_id', 'question_type', 'rate', 'errr', 'accuracy']].merge(
    L[['session_id', 'T_tired_z', 'flags', 'session_no']], on='session_id').merge(SS[['session_id', 'local_hour']], on='session_id')
D = D[~D['flags'].fillna('').str.contains('warmup') & (D.accuracy >= 0.8)].copy()      # same filters as the main pipeline (no warm-up, accuracy)
D['PP'] = D.participant_id; D = D[D.groupby('PP').PP.transform('size') >= 4].copy()
D['test'] = D.local_hour >= a.test_from
tr = D[~D.test]
mu = tr.groupby('PP')[['rate', 'errr']].mean()
sdO = {c: (tr[c] - tr.groupby('PP')[c].transform('mean')).std() for c in ['rate', 'errr']}
Zo = np.c_[(D.rate - D.PP.map(mu.rate)) / sdO['rate'], (D.errr - D.PP.map(mu.errr)) / sdO['errr']]
D['s_eff'] = Zo @ W['eff']; D['s_str'] = Zo @ W['stress']
print(f'\nOURS: {len(D)} sessions ({D.PP.nunique()} people); {int(D.test.sum())} held out (hour >= {a.test_from:g}); features centred on each person\'s TRAIN sessions')
print(f'  pooled within-person SD ours: rate {sdO["rate"]:.3f} (SWELL {sdS["rate"]:.3f}), error {sdO["errr"]:.3f} (SWELL {sdS["errr"]:.3f})')

print('\nA) HELD-OUT sessions: SWELL-model score vs our label (T_tired_z)')
te = D[D.test]; base = tr.T_tired_z.mean()
for nm in ('s_eff', 's_str'):
    pred = base + te[nm]; sk = 1 - ((pred - te.T_tired_z) ** 2).sum() / ((base - te.T_tired_z) ** 2).sum()
    print(f'  {"effort" if nm=="s_eff" else "stress"} score: skill vs train-mean {sk:+.2f}  (prediction = train-mean label + score)')
for _, r in te.iterrows():
    print(f'     {r.PP:8} {r.question_type:22} effort-score {r.s_eff:+.2f}  stress-score {r.s_str:+.2f}   actual label {r.T_tired_z:+.2f}')

print('\nB) ALL our sessions, within-person (each person centred on their own mean score and label)')
def wr(df, col, y='T_tired_z'):
    x = df[col] - df.groupby('PP')[col].transform('mean'); yy = df[y] - df.groupby('PP')[y].transform('mean'); return spearmanr(x, yy)[0]
rng = np.random.default_rng(0)
for nm, lab in (('s_eff', 'effort'), ('s_str', 'stress')):
    obs = wr(D, nm); cnt = 0
    for _ in range(a.perm):
        d2 = D.copy(); d2['T_tired_z'] = d2.groupby('PP').T_tired_z.transform(lambda s: rng.permutation(s.values)); cnt += wr(d2, nm) >= obs
    per = {p: spearmanr(x.s_eff if nm == 's_eff' else x.s_str, x.T_tired_z)[0] for p, x in D.groupby('PP')}
    print(f'  SWELL {lab} score vs label: within-person rho {obs:+.2f} (n={len(D)}), one-sided shuffle p = {cnt/a.perm:.3f};  per person: ' + ' '.join(f'{p[:3]} {v:+.2f}' for p, v in per.items()))
print('\n  component check (within-person rho vs label): ' + ', '.join(f'{c} {wr(D.assign(**{c+"_x": D[c]}), c+"_x"):+.2f}' for c in ['rate', 'errr']))
