"""Pack a run directory into a Codabench submission zip: every <rec>_desc-decomposed_events.tsv and _log.json at the zip root.
Runs the organisers' validators (validate_prediction_log / validate_prediction_file) on every pair and checks each events file
the way the format page describes it (onset, duration, sample, unit_id, event_type; zero-based integer
samples; event_type 'motor-unit-spike'; onset = sample / fs) and that every recording of the data folder is present.

usage: python src/pack_submission.py <run_dir> <out.zip> [--data data/<task>]
"""
import argparse, json, sys, zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "upstream/munitquest-algorithm-submission/bundle/scoring/src"))
from algorithm_scoring.score_recording import validate_prediction_file, validate_prediction_log  # noqa: E402

COLS = ["onset", "duration", "sample", "unit_id", "event_type"]


def check_events(tsv, fs=None):
    df = pd.read_csv(tsv, sep="\t")
    missing = [c for c in COLS if c not in df.columns]
    if missing:
        return f"missing columns {missing}"
    if len(df) == 0:
        return None
    if not np.issubdtype(df["sample"].dtype, np.integer) or (df["sample"] < 0).any():
        return "sample must be non-negative integers"
    if not np.issubdtype(df["unit_id"].dtype, np.integer):
        return "unit_id must be integers"
    if set(df["event_type"].unique()) - {"motor-unit-spike"}:
        return f"unexpected event_type values {sorted(set(df['event_type']) - {'motor-unit-spike'})}"
    if (df["duration"] != 0).any():
        return "duration must be 0"
    if fs and not np.allclose(df["onset"].to_numpy(), df["sample"].to_numpy() / fs, atol=1.0 / fs):
        return "onset does not match sample / fs"
    return None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("run_dir", type=Path); ap.add_argument("out", type=Path)
    ap.add_argument("--data", type=Path, default=None, help="data folder of the task, to check every recording is present")
    a = ap.parse_args()
    ev = sorted(a.run_dir.glob("*_desc-decomposed_events.tsv"))
    problems, n_units, n_spikes = [], 0, 0
    for f in ev:
        log = f.with_name(f.name.replace("_events.tsv", "_log.json"))
        if not log.exists():
            problems.append(f"{f.name}: no log.json"); continue
        fs = None
        try:
            meta = json.loads(log.read_text()); fs = (meta.get("Execution") or {}).get("fs") or meta.get("fs")
        except Exception as e:
            problems.append(f"{log.name}: unreadable ({e})")
        msg = check_events(f, fs)
        if msg:
            problems.append(f"{f.name}: {msg}")
        for name, (ok, errs, _) in ((log.name, validate_prediction_log(str(log))), (f.name, validate_prediction_file(str(f)))):
            if not ok:                                              # the organisers' own validators
                problems.append(f"{name}: " + "; ".join(e["issueMessage"] if isinstance(e, dict) else e.issueMessage for e in errs))
        df = pd.read_csv(f, sep="\t"); n_units += df["unit_id"].nunique(); n_spikes += len(df)
    if a.data:
        recs = sorted(p.name for p in a.data.iterdir() if p.is_dir() and p.name.startswith("recording-"))
        have = {f.name.replace("_desc-decomposed_events.tsv", "") for f in ev}
        miss = [r for r in recs if r not in have]
        if miss:
            problems.append(f"{len(miss)} recordings without output, e.g. {miss[:5]}")
    if problems:
        print("NOT PACKED:\n  " + "\n  ".join(problems)); sys.exit(1)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in ev:
            z.write(f, f.name)
            log = f.with_name(f.name.replace("_events.tsv", "_log.json")); z.write(log, log.name)
    print(f"packed {len(ev)} recordings ({n_units} units, {n_spikes} spikes) -> {a.out} ({a.out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
