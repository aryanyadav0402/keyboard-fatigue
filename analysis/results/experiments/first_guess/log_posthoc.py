"""Log the post-hoc sensitivity runs of ridge_fg5 (diagnostics.json) as one record. Run once.

    uv run --project <analysis> python log_posthoc.py
"""

import json

from candidates import N_VARIANTS
from common import HERE, TRACK, log_result
from diagnostics import POST_HOC

d = json.loads((HERE / "diagnostics.json").read_text())
log_result(TRACK, "ridge_fg5 sensitivity (post hoc, not eligible)", N_VARIANTS + len(POST_HOC),
           {"sensitivity": d["sensitivity"], "without_P5": d["without_P5"], "by_phase": d["by_phase"],
            "first_attempts": {k: d["first_attempts"][k] for k in ("spearman", "MAE", "MAE_flat")}},
           notes=f"{len(POST_HOC)} post-hoc variants (fixed alphas, leave one feature out) after the {N_VARIANTS} "
                 "registered ones; plus subset checks (without P5; check-ins 1-2 vs 3+).")
print("logged")
