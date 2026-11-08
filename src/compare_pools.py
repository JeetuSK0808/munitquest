"""Per-recording label-free comparison from pool_select tables: the old final selection (pooled first under its run name) vs the
new selection, both with the Showdown scorer's per-unit metrics on our EDF. Missing recordings count 0 (as in the assembler).

usage: python src/compare_pools.py <task_dir> <new_run> <old_run>"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
task, new, old = sys.argv[1], sys.argv[2], sys.argv[3]
rows = []
for rec_dir in sorted(p for p in (ROOT / "data" / task).iterdir() if p.is_dir() and p.name.startswith("recording-")):
    f = ROOT / "runs" / new / f"{rec_dir.name}_pool_units.csv"
    if not f.exists():
        continue
    t = pd.read_csv(f); w = t["metric_score"].fillna(0.0)
    o = t["source"] == old
    plaus = (t["cod_isi"] < 0.372) & (t["pnr"] >= 20)
    rows.append(dict(rec=rec_dir.name[10:13], old_units=int(o.sum()), old_lf=w[o].sum(), old_lf_plausible=w[o & plaus].sum(),
                     new_units=int((t["rank"] >= 0).sum()), new_scored=int(((t["rank"] >= 0) & (w > 0)).sum()), new_lf=w[t["rank"] >= 0].sum()))
d = pd.DataFrame(rows)
print(d.round(2).to_string(index=False))
print(f"{len(d)} recordings | old label-free mean {d.old_lf.mean():.3f} (plausible units only {d.old_lf_plausible.mean():.3f}) | "
      f"new {d.new_lf.mean():.3f} | new >= old on {int((d.new_lf >= d.old_lf - 1e-9).sum())}, >= old-plausible on "
      f"{int((d.new_lf >= d.old_lf_plausible - 1e-9).sum())}")
