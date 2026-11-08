"""Plausibility clean-up of an existing Showdown selection (used on isometric v2, 8 Nov 2026).

Per recording: score every written unit with the exact Showdown label-free replica (showdown_score.fast_unit_metrics: scoring
window, in-window refit, SIL / PNR / CoD-ISI) and drop the units that fail the motor-unit plausibility guard of
pool_select.choose (CoD-ISI < 0.372 and PNR >= 20 dB) wherever the scorer would credit them label-free (metric_score > 0) or
could on a slightly different preprocessing (SIL >= 0.80). Every other unit is kept unchanged and in the same order, so the
matched (label-based) part of the file is untouched apart from the dropped noise units.

usage: .venv_score/bin/python src/guard_filter.py <task> <in_run> <out_run> [--recs 001,002] [--sil-margin 0.80]"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pool_select import PLAUS_COD, PLAUS_PNR, load_run_units, write_tsv  # noqa: E402
from proxy_score import bandpassed_edf  # noqa: E402
from showdown_score import fast_unit_metrics, mode_of, window  # noqa: E402

RULE = ("units that are not motor-unit-like (CoD-ISI >= 0.372 or PNR < 20 dB) dropped where the Showdown scorer would credit "
        "them label-free or their SIL is >= {m:.2f}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("task"); ap.add_argument("inp", type=Path); ap.add_argument("out", type=Path)
    ap.add_argument("--recs", default=""); ap.add_argument("--sil-margin", type=float, default=0.80)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    from muniverse.algorithms.post_processing import PostProcessCBSS
    from pyedflib.highlevel import read_edf
    recs = sorted(p for p in (ROOT / "data" / a.task).iterdir() if p.is_dir() and p.name.startswith("recording-"))
    if a.recs:
        want = {f"recording-{r.zfill(3)}" for r in a.recs.split(",")}; recs = [p for p in recs if p.name.split("_")[0] in want]
    cache = ROOT / "cache" / a.task
    for rec_dir in recs:
        rec = rec_dir.name; t0 = time.time()
        if not (a.inp / f"{rec}_desc-decomposed_events.tsv").exists():
            print(f"{rec}: no input", flush=True); continue
        if (a.out / f"{rec}_desc-decomposed_events.tsv").exists():
            print(f"{rec}: skip", flush=True); continue
        fs = float(json.load(open(rec_dir / f"{rec}_emg.json"))["SamplingFrequency"]); w0, w1 = window(rec_dir)
        units = load_run_units(a.inp, rec)
        sig, _, _ = read_edf(str(bandpassed_edf(rec_dir, cache))); pp = PostProcessCBSS(steps=[], verbose=False)
        X = pp._whitening(pp._extension(np.asarray(sig, np.float64))); del sig
        m = fast_unit_metrics(X, units, fs, mode_of(rec), w0, w1); del X
        plaus = ((m["cod_isi"] < PLAUS_COD) & (m["pnr"] >= PLAUS_PNR)).to_numpy()
        drop = m["sil"].notna().to_numpy() & ~plaus & ((m["metric_score"] > 0) | (m["sil"] >= a.sil_margin)).to_numpy()
        m.insert(0, "unit_id", np.arange(len(units))); m.insert(1, "n_spikes", [len(S) for S in units])
        m["plausible"] = plaus; m["dropped"] = drop
        m.to_csv(a.out / f"{rec}_guard_units.csv", index=False)
        kept = [S for S, d in zip(units, drop) if not d]
        old_sum, new_sum = float(m["metric_score"].sum()), float(m.loc[~drop, "metric_score"].sum())
        log = json.load(open(a.inp / f"{rec}_desc-decomposed_log.json"))
        g = log["GeneratedBy"][0]
        g["Description"] = g["Description"].rstrip(".") + "; then " + RULE.format(m=a.sil_margin) + "."
        g["Version"] = "0.3"
        ex = log["Execution"]; up = float(ex.get("Runtime", 0.0))
        ex["guard_filter"] = dict(source_run=str(a.inp), units_in=len(units), units_dropped=int(drop.sum()),
                                  rule=RULE.format(m=a.sil_margin), runtime_s=round(time.time() - t0, 1))
        ex["written_units"] = len(kept); ex["Runtime"] = float(round(up + time.time() - t0, 1))
        if kept:
            write_tsv(a.out / f"{rec}_desc-decomposed_events.tsv", kept, fs)
            json.dump(log, open(a.out / f"{rec}_desc-decomposed_log.json", "w"), indent=1)
        print(f"{rec}: {len(units)} units -> {len(kept)} kept ({int(drop.sum())} dropped, {int((drop & (m['metric_score'] > 0).to_numpy()).sum())} "
              f"credited) | label-free {old_sum:.2f} -> {new_sum:.2f} | {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
