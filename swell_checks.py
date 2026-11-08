#!/usr/bin/env python3
"""Sanity checks on the SWELL two-stage model, on OUR data, same three splits as swell_stack.py:
 A) latest session per person held out   B) leave-one-session-out   C) leave-one-person-out
Predictors for the tiny 3-number model (intercept + 1-2 weights, ridge):
  clock only        = person-centred session number              (is it just 'later session = more tired'?)
  raw rate+backspace= person-centred typing rate and backspace ratio, no SWELL
  backspace only    = person-centred backspace ratio
  SWELL stress/effort scores (stage 1 trained on SWELL only)
  clock + each of the above (does it add anything once practice is accounted for?)
  python3 swell_checks.py --swell swell.pkl --out out --test-from 22"""
import argparse, io, contextlib, runpy, sys, numpy as np, pandas as pd
from scipy.stats import spearmanr
ap = argparse.ArgumentParser(); ap.add_argument('--swell', required=True); ap.add_argument('--out', default='out')
ap.add_argument('--test-from', type=float, default=22.0); ap.add_argument('--alpha', type=float, default=1.0)
ap.add_argument('--train-before', type=float, default=None, help='fit only on sessions before this hour (default = --test-from)')
a = ap.parse_args()
sys.argv = ['swell_to_ours.py', '--swell', a.swell, '--out', a.out, '--test-from', str(a.test_from), '--perm', '1']
with contextlib.redirect_stdout(io.StringIO()):
    ns = runpy.run_path('swell_to_ours.py')
D, W = ns['D'].reset_index(drop=True), ns['W']
n = len(D); y = D.T_tired_z.values; pp = D.PP.values
D['left_err'] = 1 - D.accuracy
def feats(centre):
    """all candidate predictors; each person's mean (and the pooled SD) come only from rows where centre is True"""
    cols = ['rate', 'errr', 'left_err']; X = D[cols].copy(); X['clock'] = D.session_no.astype(float)
    mu = D[centre].assign(clock=D.session_no.astype(float)).groupby('PP')[cols + ['clock']].mean()
    Z = pd.DataFrame({c: (X[c] - D.PP.map(mu[c])) for c in cols + ['clock']})
    Z = Z / pd.Series({c: Z.loc[centre, c].std() + 1e-9 for c in Z.columns})
    out = {c: Z[c].values for c in Z.columns}
    sw = np.c_[Z.rate.values, Z.errr.values]
    out['swell_eff'], out['swell_str'] = sw @ W['eff'], sw @ W['stress']
    return out
def design(F, names): return np.c_[tuple(F[k] for k in names)]
def fit(Xtr, ytr):
    m = Xtr.mean(0); s = Xtr.std(0) + 1e-9; Z = (Xtr - m) / s; mu = ytr.mean()
    w = np.linalg.solve(Z.T @ Z + a.alpha * np.eye(Z.shape[1]), Z.T @ (ytr - mu))
    return lambda X: ((X - m) / s) @ w + mu
def skill(p, act, b): return 1 - ((p - act) ** 2).sum() / ((b - act) ** 2).sum()
models = {
 'clock only (session number)':           ['clock'],
 'raw rate + backspace (no SWELL)':       ['rate', 'errr'],
 'backspace only (no SWELL)':             ['errr'],
 'SWELL stress score':                    ['swell_str'],
 'SWELL effort + stress scores':          ['swell_eff', 'swell_str'],
 'clock + raw rate + backspace':          ['clock', 'rate', 'errr'],
 'clock + SWELL stress score':            ['clock', 'swell_str'],
}
te = (D.local_hour >= a.test_from).values; trA = (D.local_hour < (a.train_before if a.train_before is not None else a.test_from)).values
Fa = feats(trA); F0 = feats(np.ones(n, bool))
res = {k: {} for k in models}
for k, cols in models.items():
    f = fit(design(Fa, cols)[trA], y[trA]); pr = f(design(Fa, cols)[te]); res[k]['A'] = (skill(pr, y[te], np.full(te.sum(), y[trA].mean())), pr)
    pred = np.zeros(n); base = np.zeros(n)
    for i in range(n):
        tr = np.arange(n) != i; Fi = feats(tr); pred[i] = fit(design(Fi, cols)[tr], y[tr])(design(Fi, cols)[[i]])[0]; base[i] = y[tr].mean()
    res[k]['B'] = (skill(pred, y, base), spearmanr(pred, y)[0])
    pred = np.zeros(n); base = np.zeros(n)
    for p in sorted(set(pp)):
        tp = pp == p; pred[tp] = fit(design(F0, cols)[~tp], y[~tp])(design(F0, cols)[tp]); base[tp] = y[~tp].mean()
    res[k]['C'] = (skill(pred, y, base), spearmanr(pred, y)[0])
print(f'{n} sessions, {len(set(pp))} people; held-out = {int(te.sum())} latest sessions (hour >= {a.test_from:g}); fitted on {int(trA.sum())} sessions before {(a.train_before if a.train_before is not None else a.test_from):g}:00')
print(f'\n{"model (intercept + weights, ridge)":38} {"A hold-out":>10} {"B 1-session-out":>20} {"C 1-person-out":>20}')
for k in models:
    A_, B_, C_ = res[k]['A'][0], res[k]['B'], res[k]['C']
    print(f'{k:38} {A_:+10.2f} {B_[0]:+11.2f} (rho {B_[1]:+.2f}) {C_[0]:+11.2f} (rho {C_[1]:+.2f})')
print('\nheld-out predictions (predicted / actual):  ' + '  '.join(f'{p[:3]} {a_:+.2f}' for p, a_ in zip(pp[te], y[te])))
for k in models: print(f'   {k:38} ' + '  '.join(f'{p:+.2f}' for p in res[k]['A'][1]))

print('\nIs it practice? within-person rank correlations over our sessions (n=%d):' % n)
def wr(x, z): 
    xc = pd.Series(x) - pd.Series(x).groupby(pp).transform('mean'); zc = pd.Series(z) - pd.Series(z).groupby(pp).transform('mean'); return spearmanr(xc, zc)[0]
ci = lambda c: D[c].values
print(f'   session number vs tiredness label        {wr(D.session_no.values, y):+.2f}')
print(f'   session number vs backspace ratio        {wr(D.session_no.values, ci("errr")):+.2f}   (negative = backspaces fall with practice)')
print(f'   backspace ratio vs tiredness label       {wr(ci("errr"), y):+.2f}')
D['left_err'] = 1 - D.accuracy
print(f'   errors LEFT IN (1 - final accuracy) vs label {wr(ci("left_err"), y):+.2f}   ({int((D.left_err>0).sum())} of {n} sessions have any)')
print(f'   left-in minus fixes (z) vs label         {wr((D.left_err/D.left_err.std()).values - (D.errr/D.errr.std()).values, y):+.2f}')
# does backspace still relate to the label after removing the session-number trend inside each person?
res_ = []
for v in ('errr', 'rate'):
    r = pd.Series(ci(v), index=D.index).copy()
    for p in set(pp):
        m = pp == p; xs = D.session_no.values[m].astype(float); 
        if m.sum() >= 3: b = np.polyfit(xs, r[m].values, 1); r[m] = r[m].values - np.polyval(b, xs)
    res_.append((v, wr(r.values, y)))
print('   after removing each person\'s own practice trend (session number) from the feature:  ' + ', '.join(f'{v} vs label {r:+.2f}' for v, r in res_))
