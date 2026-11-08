"""Motor-unit filter refinement in the scorer's own space, then best-first ordering.

For every predicted unit: whiten the 12x-extended recording exactly as the official post-processing does (PostProcessCBSS,
ZCA, extension 12), then iterate  w = mean of whitened samples at the spikes (unit norm) -> source = w.X -> spikes = peak
detection + 2-means on the source (muniverse est_spike_times, a = 2, 10 ms min distance)  until the spike train stops changing.
This is the standard spike-triggered filter refinement (as in CKC / CBSS decompositions); it makes each spike train consistent
with the filter the scorer rebuilds from it. A unit keeps its refined train only if the scorer's SIL measure improves; units are
then written best-first by SIL so the scorer's "keep first" duplicate rule keeps the cleaner copy.

usage: python src/refine_units.py <task> <in_run_dir> <out_run_dir> [--recs 001,002] [--iters 8]"""
import argparse, json, shutil, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from pyedflib.highlevel import read_edf
from muniverse.algorithms.post_processing import PostProcessCBSS
from muniverse.algorithms.core import est_spike_times
from muniverse.evaluation.evaluate import pseudo_sil_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from proxy_score import bandpassed_edf  # noqa: E402
from run_cbss import common_fraction  # noqa: E402


def dedup_merge(cands, fs, thr=0.3):
    """cands: list of (refined_sil, refined_spikes, orig_sil, orig_spikes). Best-first by refined SIL; keep the refined train unless
    it duplicates a kept unit (scorer rule: >= 30 % shared spikes, 1 ms tolerance, 10 ms shift), then fall back to the original
    train if that one is distinct. Refinement tends to pull weak units onto the same strong one, so this keeps the diversity."""
    tol = max(1, int(round(0.001 * fs))); ms = int(round(0.01 * fs)); kept = []
    dup = lambda S: any(common_fraction(S, k[1], tol, ms) >= thr for k in kept)
    for r_sil, r_S, o_sil, o_S in sorted(cands, key=lambda c: -(c[0] if np.isfinite(c[0]) else -1)):
        for sil, S in ((r_sil, r_S), (o_sil, o_S)):
            if len(S) >= 10 and np.isfinite(sil) and not dup(S):
                kept.append((sil, S)); break
    return kept


def refine(X, spikes, fs, iters):
    def sil_of(S):
        w = X[:, S].mean(1); w /= np.linalg.norm(w) + 1e-12; src = w @ X
        return pseudo_sil_score(source=src, spikes=S, fsamp=fs, min_peak_dist=0.01, match_dist=0.001)[0], src
    S = np.unique(spikes[(spikes >= 0) & (spikes < X.shape[1])])
    if len(S) < 10:
        return S, np.nan, np.nan
    sil0, src = sil_of(S); best_S, best_sil = S, sil0
    for _ in range(iters):
        out = est_spike_times(source=src, fsamp=fs, a=2, min_delay=0.01)
        S_new = np.asarray(out[0] if isinstance(out, tuple) else out, dtype=np.int64)
        if len(S_new) < 10:
            break
        sil, src = sil_of(S_new)
        if sil > best_sil:
            best_S, best_sil = S_new, sil
        if len(S_new) == len(S) and np.array_equal(S_new, S):
            break
        S = S_new
    return best_S, sil0, best_sil


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("task"); ap.add_argument("inp", type=Path); ap.add_argument("out", type=Path)
    ap.add_argument("--recs", default=""); ap.add_argument("--iters", type=int, default=8)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    cache = ROOT / "cache" / a.task; cache.mkdir(parents=True, exist_ok=True)
    preds = sorted(a.inp.glob("*_desc-decomposed_events.tsv"))
    if a.recs:
        keep = set(a.recs.split(",")); preds = [p for p in preds if any(f"recording-{r}" in p.name for r in keep)]
    for p in preds:
        t0 = time.time(); rec = p.name.split("_desc-")[0]
        rec_dir = next((ROOT / "data").rglob(f"{rec}*"), None)
        rec_dir = rec_dir if rec_dir.is_dir() else rec_dir.parent
        edf = bandpassed_edf(rec_dir, cache)
        sig, heads, _ = read_edf(str(edf)); sig = np.asarray(sig, dtype=np.float64); fs = float(heads[0]["sample_frequency"])
        pp = PostProcessCBSS(steps=[], verbose=False) if "verbose" in PostProcessCBSS.__init__.__code__.co_varnames else PostProcessCBSS(steps=[])
        X = pp._whitening(pp._extension(sig))
        df = pd.read_table(p); cands, stats = [], []
        for u, g in df.groupby("unit_id"):
            S0 = np.unique(g["sample"].to_numpy(np.int64)); S, s0, s1 = refine(X, S0, fs, a.iters)
            stats.append((u, s0, s1, len(g), len(S))); cands.append((s1, S, s0, S0))
        rows = dedup_merge(cands, fs)
        out = [(S_i / fs, 0, int(S_i), k, "motor-unit-spike") for k, (_, S) in enumerate(rows) for S_i in S]
        pd.DataFrame(out, columns=["onset", "duration", "sample", "unit_id", "event_type"]).to_csv(a.out / p.name, sep="\t", index=False, float_format="%.6f")
        log = p.with_name(p.name.replace("_events.tsv", "_log.json"))
        if log.exists():
            shutil.copy(log, a.out / log.name)
        st = np.array([(s0, s1) for _, s0, s1, _, _ in stats if np.isfinite(s0)])
        print(f"{rec}: units {len(stats)} -> distinct {len(rows)} | SIL>=0.9 before {int((st[:, 0] >= 0.9).sum())} after {int((st[:, 1] >= 0.9).sum())} | "
              f"mean SIL {st[:, 0].mean():.3f} -> {st[:, 1].mean():.3f} | {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()



def wrefine(X, spikes, fs, t0, t1, iters=8):
    """Window-consistent refinement for the Showdown scorer. Its refit builds the filter from the spikes inside the scoring window
    only (mean of the whitened samples there, whitening on the whole recording), and in dynamic mode it re-detects spikes on that
    source (est_spike_times, defaults) and scales the unit's score by the share of them the unit predicts. Iterate filter from the
    in-window spikes -> source -> est_spike_times until the train stops changing, so the train is that refit's own fixed point."""
    lo, hi = t0 * fs, t1 * fs
    S = np.unique(spikes[(spikes >= 0) & (spikes < X.shape[1])])
    for _ in range(iters):
        Sin = S[(S > lo) & (S < hi)]
        if len(Sin) < 10:
            break
        w = X[:, Sin].mean(1); w /= np.linalg.norm(w) + 1e-12
        out = est_spike_times(source=w @ X, fsamp=fs)
        S_new = np.asarray(out[0] if isinstance(out, tuple) else out, dtype=np.int64)
        if len(S_new) < 10 or np.array_equal(S_new, S):
            break
        S = S_new
    return S
