import json, time, warnings
warnings.filterwarnings("ignore")
from common import h
from common import D
import exp2, exp3
f = exp3.CANDS["win24 ridge (abs)"]
t0 = time.time()
obs, p, null = h.perm_test(D, f, "lopo", "skill", n=100)
print("perm lopo skill obs %.3f p %.3f null mean %.3f sd %.3f 95%% %.3f  (%.0fs)" % (obs, p, null.mean(), null.std(), __import__('numpy').percentile(null, 95), time.time() - t0))
obs2, p2, null2 = h.perm_test(D, f, "rolling", "skill", n=100)
print("perm rolling skill obs %.3f p %.3f null mean %.3f" % (obs2, p2, null2.mean()))
json.dump({"lopo": [obs, p], "rolling": [obs2, p2]}, open("perm_final.json", "w"))
# log everything
for fn in ("exp1", "exp2", "exp3"):
    pass
