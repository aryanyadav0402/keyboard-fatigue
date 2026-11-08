"""Sliding-window models: ridge / MLP / 1D CNN / GRU on windows of baseline-centred log-IKI, log-hold, backspace flag."""
import json, sys, warnings
warnings.filterwarnings("ignore")
import torch, torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from common import *
torch.set_num_threads(1)
pid_of = dict(zip(T.checkin_id, T.pid))
def pool(p, k): return np.concatenate([SEQ[c][k] for c in BASE[p]])
MED = {p: (np.median(pool(p, 0)), np.median(pool(p, 1))) for p in BASE}
SD = np.std(np.concatenate([SEQ[c][0] - MED[pid_of[c]][0] for p in BASE for c in BASE[p]])), \
     np.std(np.concatenate([SEQ[c][1] - MED[pid_of[c]][1] for p in BASE for c in BASE[p]]))
def channels(c):  # (L,3) centred, scaled
    a, b, bs = SEQ[c]; m = MED[pid_of[c]]
    return np.stack([(a - m[0]) / SD[0], (b - m[1]) / SD[1], bs], 1).astype(np.float32)
CH = {c: channels(c) for c in SEQ if pid_of[c] in BASE}
def windows(c, W, S):
    x = CH[c]; return np.stack([x[i:i + W] for i in range(0, len(x) - W + 1, S)])

def build(frame, W, S, anch, tgt=True):
    X, y, g = [], [], []
    for k, (_, r) in enumerate(frame.iterrows()):
        w = windows(r.checkin_id, W, S); X.append(w); g += [k] * len(w)
        y += [(r.kss_delta if anch else r.kss)] * len(w)
    return np.concatenate(X), np.array(y, float), np.array(g)
def agg(p, g, n): return np.array([p[g == k].mean() for k in range(n)])
def summ(X):  # per-window summary stats
    return np.concatenate([X.mean(1), X.std(1), X.max(1), X.min(1)], 1)

class Net(nn.Module):
    def __init__(s, kind, hid=8, p=0.3):
        super().__init__(); s.kind = kind
        if kind == "cnn":
            s.f = nn.Sequential(nn.Conv1d(3, hid, 3, padding=1), nn.ReLU(), nn.Dropout(p), nn.Conv1d(hid, hid, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1), nn.Flatten())
        else:
            s.g = nn.GRU(3, hid, batch_first=True)
        s.o = nn.Sequential(nn.Dropout(p), nn.Linear(hid, 1))
    def forward(s, x):
        if s.kind == "cnn": z = s.f(x.transpose(1, 2))
        else: z = s.g(x)[1][0]
        return s.o(z).squeeze(-1)

def fit_torch(kind, X, y, Xt, seeds=3, epochs=30, wd=0.1):
    mu, sd = y.mean(), y.std() + 1e-9; yt = torch.tensor((y - mu) / sd, dtype=torch.float32)
    Xa, Xb = torch.tensor(X), torch.tensor(Xt); preds = []
    for s in range(seeds):
        torch.manual_seed(s); net = Net(kind); opt = torch.optim.AdamW(net.parameters(), 3e-3, weight_decay=wd)
        for ep in range(epochs):
            net.train(); perm = torch.randperm(len(Xa))
            for i in range(0, len(Xa), 32):
                b = perm[i:i + 32]; opt.zero_grad(); nn.functional.mse_loss(net(Xa[b]), yt[b]).backward(); opt.step()
        net.eval()
        with torch.no_grad(): preds.append(net(Xb).numpy())
    return np.mean(preds, 0) * sd + mu

def make(kind, anch, W=16, S=4, alpha=None):
    def fp(train, test):
        X, y, g = build(train, W, S, anch); Xt, _, gt = build(test, W, S, anch)
        if kind == "ridge":
            Xf, Xtf = X.reshape(len(X), -1), Xt.reshape(len(Xt), -1)
            p = Ridge(alpha=alpha or 1000).fit(Xf, y).predict(Xtf)
        elif kind == "summ":
            p = Ridge(alpha=alpha or 100).fit(summ(X), y).predict(summ(Xt))
        elif kind == "mlp":
            ym = y.mean(); p = ym + np.mean([MLPRegressor(hidden_layer_sizes=(8,), alpha=10, max_iter=300, early_stopping=False, random_state=s).fit(X.reshape(len(X), -1), y - ym).predict(Xt.reshape(len(Xt), -1)) for s in range(3)], 0)
        else:
            p = fit_torch(kind, X, y, Xt)
        p = agg(p, gt, len(test))
        return p + (test.kss_base.to_numpy(float) if anch else 0)
    return fp

CANDS = {}
for anch in (True, False):
    tag = "anch" if anch else "abs"
    for kind in ("ridge", "summ", "mlp", "cnn", "gru"):
        CANDS[f"win16 {kind} ({tag})"] = make(kind, anch)
if __name__ == "__main__":
    out = {}
    for n, f in CANDS.items():
        r = h.evaluate(D, f); out[n] = r
        print(f"{n:28s} lopo MAE {r['lopo']['MAE']:.3f} skill {r['lopo']['skill']:+.3f} conc {r['lopo']['within_concordance']:.3f} | roll MAE {r['rolling']['MAE']:.3f} skill {r['rolling']['skill']:+.3f} conc {r['rolling']['within_concordance']:.3f}", flush=True)
    json.dump(out, open("exp2_results.json", "w"), default=float)
