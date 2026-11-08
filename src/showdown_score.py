"""Local replica of the Showdown scorer's label-free branch (organisers' scoring code, MUnitQuest/munitquest-algorithm-submission
46bdda9, run with muniverse-emg 0.1.7 in .venv_score). Per recording: only the window between the first `scoring_start` and
`scoring_end` events counts; duplicates go first (+-10 ms, 30 % of the longer train); then the CBSS refit from the in-window
spikes (whitening on the whole recording) gives SIL / PNR / CoD-ISI, binned (SIL 0.85 / 0.925 / 0.95, PNR 28 / 33.9 / 37.1 dB);
units need SIL >= 0.85 and >= 25 in-window spikes; in dynamic mode each unit's score is also multiplied by the "model recall"
(share of the spikes re-detected on the refit source that the unit predicts; <= 0.5 -> 0). With no labels every unit is
unmatched, so this is the score of the incomplete-label recordings; full-label recordings are scored on matched F1 only.
Our 20-500 Hz band-passed EDF stands in for the organisers' (unpublished) preprocessed one.

usage: .venv_score/bin/python src/showdown_score.py <task_dir> <pred_dir> [--recs 001,002]
       (mean over every recording of data/<task_dir>; a missing prediction counts 0, as in the official assembler)"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "upstream/munitquest-algorithm-submission/bundle/scoring/src"))
sys.path.insert(0, str(ROOT / "src"))
from algorithm_scoring.score_recording import MUnitQuestScoring, ScoringConfig  # noqa: E402

EMPTY_GT = ROOT / "cache" / "empty_groundtruth_events.tsv"


def window(rec_dir: Path):
    ev = pd.read_table(next(rec_dir.glob("*_events.tsv")))
    return (float(ev.loc[ev["event_type"] == "scoring_start", "onset"].values[0]),
            float(ev.loc[ev["event_type"] == "scoring_end", "onset"].values[0]))


def mode_of(rec: str):
    return "dynamic" if "_challenge-dynamic" in rec else "isometric"


def steps(t0, t1, dedup=True):
    s = [{"step": "remove_duplicates", "max_shift": 0.01, "tolerance": 0.001, "threshold": 0.3, "mode": "first"}] if dedup else []
    return s + [{"step": "validate_prediction", "t_start": t0, "t_end": t1, "threshold": 0.5},
                {"step": "fit_from_spikes", "t_start": t0, "t_end": t1},
                {"step": "get_discharge_metric", "metric": "cod_isi"}]


def score_file(pred_tsv, edf, fs, mode, t0, t1, dedup=True):
    if not EMPTY_GT.exists():
        EMPTY_GT.parent.mkdir(parents=True, exist_ok=True); EMPTY_GT.write_text("onset\tduration\tsample\tunit_id\tevent_type\n")
    cfg = ScoringConfig(fsamp=fs, steps=steps(t0, t1, dedup), data=str(edf), ground_truth=str(EMPTY_GT), signal_metrics=True)
    s = MUnitQuestScoring(prediction=str(pred_tsv), cfg=cfg)
    s.get_score(mode=mode, t_start=t0, t_end=t1)
    return s.metrics, s.unit_metrics


def main():
    from proxy_score import bandpassed_edf
    ap = argparse.ArgumentParser(); ap.add_argument("task"); ap.add_argument("pred", type=Path); ap.add_argument("--recs", default="")
    ap.add_argument("--band", default="20,500", help="band-pass of the stand-in EDF (robustness check: e.g. 10,500)")
    ap.add_argument("--tag", default="", help="suffix for the output csv files")
    a = ap.parse_args(); band = tuple(int(x) for x in a.band.split(","))
    cache = ROOT / "cache" / a.task; cache.mkdir(parents=True, exist_ok=True)
    recs = sorted(p for p in (ROOT / "data" / a.task).iterdir() if p.is_dir() and p.name.startswith("recording-"))
    if a.recs:
        want = {f"recording-{r.zfill(3)}" for r in a.recs.split(",")}; recs = [p for p in recs if p.name.split("_")[0] in want]
    rows = []
    for rec_dir in recs:
        rec = rec_dir.name; p = a.pred / f"{rec}_desc-decomposed_events.tsv"
        if not p.exists():
            rows.append(dict(recording=rec, score=0.0, accepted_units=0)); print(f"{rec}: no prediction -> 0", flush=True); continue
        fs = float(json.load(open(rec_dir / f"{rec}_emg.json"))["SamplingFrequency"]); t0, t1 = window(rec_dir)
        m, units = score_file(p, bandpassed_edf(rec_dir, cache, band), fs, mode_of(rec), t0, t1)
        units.to_csv(a.pred / f"{rec}_showdown_units{a.tag}.csv", index=False)
        rows.append(dict(recording=rec, **{k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v) for k, v in m.items()}))
        print(f"{rec}: score {m['score']:.2f} | accepted {m.get('accepted_units')} rejected {m.get('rejected_units')} "
              f"| recall {m.get('recall_mean', float('nan')):.3f}", flush=True)
    df = pd.DataFrame(rows); df.to_csv(a.pred / f"showdown_scores{a.tag}.csv", index=False)
    print(f"MEAN label-free Showdown score over {len(df)} recordings: {df['score'].mean():.3f}")


if __name__ == "__main__":
    main()


# ---- fast exact replica of the per-unit computation (same formulas as muniverse 0.1.7 fit_from_spikes + the Showdown get_score;
# ---- the 41 delays of a unit share one matrix product). Checked against score_file on real units (see HANDOFF, 7 Nov).
SIL_BINS, PNR_BINS, COD_BINS = (0.85, 0.925, 0.95), (28, 33.9, 37.1), (0.05, 0.197, 0.372, 0.730)


def _map(x, bins, vals):
    if not np.isfinite(x):
        return 0.0
    for b, v in zip(bins, vals):
        if x < b:
            return v
    return vals[len(bins)]


def sil_map(s):
    return 0.0 if not np.isfinite(s) or s > 1 else _map(s, SIL_BINS, (0, 0.5, 0.75, 1))


def pnr_map(p):
    return _map(p, PNR_BINS, (0, 0.5, 0.75, 1))


def cod_map(c):
    return _map(c, COD_BINS, (0, 1, 0.75, 0.5, 0))


def _pseudo_sil(sig, peaks, spikes, mw):
    """muniverse pseudo_sil_score on sig = source*|source| with precomputed peaks; spikes sorted unique."""
    if len(spikes) < 2:
        return 0.0
    i = np.searchsorted(spikes, peaks); lo = spikes[np.clip(i - 1, 0, len(spikes) - 1)]; hi = spikes[np.clip(i, 0, len(spikes) - 1)]
    back = peaks[(np.abs(peaks - lo) > mw) & (np.abs(peaks - hi) > mw)]
    if len(back) == 0:
        return 0.0
    p = sig[spikes]; c0, c1 = p.mean(), sig[back].mean()
    within = ((p - c0) ** 2).sum() if len(p) > 1 else 0.0; between = ((p - c1) ** 2).sum(); m = max(between, within)
    return float((between - within) / m) if m > 0 else 0.0


def _pnr(source, spikes):
    sig = source * np.abs(source); n = len(sig); m = np.zeros(n, bool)
    for d in (-1, 0, 1):
        j = spikes + d; m[j[(j >= 0) & (j < n)]] = True
    return float(10 * np.log10(np.mean(sig[m] ** 2) / np.mean(sig[~m] ** 2)))


def fast_unit_metrics(X, trains, fs, mode, t0, t1, max_delay=0.01, min_fit=10, min_n=25):
    """Per-unit Showdown label-free metrics for every train (no duplicate step), as score_file(dedup=False) gives them.
    X = PostProcessCBSS whitened, extended full recording. Returns a DataFrame (sil, pnr, cod_isi, recall, metric_score)."""
    from scipy.signal import find_peaks
    from scipy.stats import median_abs_deviation
    from muniverse.algorithms.core import est_spike_times
    from muniverse.evaluation.evaluate import bin_spikes, max_xcorr, match_spike_trains
    N = X.shape[1]; ms = int(max_delay * fs); delays = np.arange(-ms, ms + 1); mw = int(round(fs * 0.001)); md = int(round(fs * 0.01))
    rows = []
    for S in trains:
        on = S / fs
        n_in = int(((on >= t0) & (on < t1)).sum())                      # validate_prediction window (FP count vs empty labels)
        loc = S[(S > t0 * fs) & (S < t1 * fs)].astype(int)              # fit_from_spikes window
        r = dict(n_in=n_in, sil=np.nan, pnr=np.nan, cod_isi=np.nan, recall=np.nan, metric_score=0.0)
        if len(loc) >= min_fit and loc.min() - ms >= 0 and loc.max() + ms < N:
            W = np.stack([X[:, loc + d].mean(1) for d in delays], 1); W /= np.linalg.norm(W, axis=0, keepdims=True)
            src = W.T @ X; sils = np.empty(len(delays))
            for k in range(len(delays)):
                sig = src[k] * np.abs(src[k]); pk, _ = find_peaks(sig, distance=md)
                sils[k] = _pseudo_sil(sig, pk, loc + delays[k], mw)
            k = int(np.argmax(sils)); sp = loc + delays[k]; source = src[k]
            isi = np.diff(sp / fs)
            r.update(sil=float(sils[k]), pnr=_pnr(source, sp),
                     cod_isi=float(median_abs_deviation(isi) / np.median(isi)) if len(isi) > 1 else np.nan)
            ok = r["sil"] >= SIL_BINS[0] and n_in >= min_n
            score = (sil_map(r["sil"]) + pnr_map(r["pnr"]) + cod_map(r["cod_isi"])) / 3 if ok else 0.0
            if mode == "dynamic" and ok:
                est, _ = est_spike_times(source=source, fsamp=fs); est = np.asarray(est)
                est = est[(est > t0 * fs) & (est < t1 * fs)]
                pred = on[(on > t0) & (on < t1)]
                a = bin_spikes(spike_times=pred, fsamp=fs, t_start=t0, t_end=t1)
                b = bin_spikes(spike_times=est / fs, fsamp=fs, t_start=t0, t_end=t1)
                _, shift = max_xcorr(a, b, max_shift=int(0.1 * fs))
                tp, fp, fn = match_spike_trains(a, b, shift=shift, fsamp=fs)
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0; rec = 0.0 if rec <= 0.5 else rec
                r["recall"] = rec; score *= rec
            r["metric_score"] = float(score)
        rows.append(r)
    return pd.DataFrame(rows)
