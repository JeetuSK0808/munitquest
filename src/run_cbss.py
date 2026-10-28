"""MUnitQuest decomposition with MUniverse FastIcaCBSS (one or more runs per recording, merged), written as competition files.

usage: python src/run_cbss.py <task> <out_dir> [--recs 001,002] [--iters 80] [--ext 12] [--seeds 1909] [--threads 2]
Per recording: <rec>_desc-decomposed_events.tsv (units ordered best-first, duplicates removed with the scorer's own rule)
and <rec>_desc-decomposed_log.json (GeneratedBy / RuntimeEnvironment / Execution)."""
import argparse, json, platform, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def bandpass(X, fs):
    from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt
    Y = sosfiltfilt(butter(4, [20, 500], btype="band", fs=fs, output="sos"), X.astype(np.float64), axis=0)
    return Y


def spikes_by_unit(df):
    return {int(u): np.sort(g["sample"].to_numpy(dtype=np.int64)) for u, g in df.groupby("unit_id")}


def common_fraction(a, b, tol, max_shift):
    """Best fraction of common spikes (relative to the smaller train) over integer lags within +-max_shift samples."""
    if len(a) == 0 or len(b) == 0:
        return 0.0
    best = 0.0
    for lag in range(-max_shift, max_shift + 1, max(1, tol)):
        bb = b + lag; idx = np.searchsorted(bb, a); idx = np.clip(idx, 1, len(bb) - 1)
        d = np.minimum(np.abs(bb[idx] - a), np.abs(bb[idx - 1] - a)); c = (d <= tol).sum() / min(len(a), len(b))
        best = max(best, c)
    return best


def merge_units(units, fs, thr=0.3):
    """units: list of (quality, spikes). Keep best-first, dropping any unit sharing >= thr of spikes with a kept one
    (scorer: max_shift 10 ms, tolerance 1 ms, threshold 0.3, keep first)."""
    tol = max(1, int(round(0.001 * fs))); ms = int(round(0.01 * fs))
    kept = []
    for q, s in sorted(units, key=lambda t: -t[0]):
        if len(s) < 10:
            continue
        if all(common_fraction(s, k[1], tol, ms) < thr for k in kept):
            kept.append((q, s))
    return kept


def cov_isi(s, fs):
    isi = np.diff(np.sort(s)) / fs
    return float(isi.std() / isi.mean()) if len(isi) > 2 and isi.mean() > 0 else np.inf


def refine_all(Y, units, fs, iters=8, min_sil=0.9):
    """Spike-triggered filter refinement of every candidate in the scorer's own space (PostProcessCBSS whitening, extension 12;
    see src/refine_units.py), then drop candidates whose refined SIL is below the scorer's 0.9 cut so they cannot knock a good
    duplicate out under the scorer's keep-first rule. Quality = refined SIL, lightly penalised for irregular firing."""
    from muniverse.algorithms.post_processing import PostProcessCBSS
    from refine_units import refine
    pp = PostProcessCBSS(steps=[], verbose=False); Xw = pp._whitening(pp._extension(Y))
    out = []
    for _, s in units:
        S, _, s1 = refine(Xw, np.asarray(s, np.int64), fs, iters)
        if np.isfinite(s1) and s1 >= min_sil:
            out.append((s1 - 0.1 * max(0.0, min(cov_isi(S, fs), 2.0) - 0.3), S))
    del Xw
    return out


def write(out_dir, rec, kept, fs, meta):
    rows = []
    for u, (q, s) in enumerate(kept):
        rows += [(int(x) / fs, 0, int(x), u, "motor-unit-spike") for x in s]
    rows.sort(key=lambda r: (r[2], r[3]))
    with open(out_dir / f"{rec}_desc-decomposed_events.tsv", "w") as f:
        f.write("onset\tduration\tsample\tunit_id\tevent_type\n")
        for o, d, s, u, e in rows:
            f.write(f"{o:.6f}\t{d}\t{s}\t{u}\t{e}\n")
    log = {"GeneratedBy": [{"Name": "BHopper-MUniverseCBSS-ensemble", "Description": "MUniverse FastIcaCBSS decompositions (one or more seeds) "
                            "merged best-first with the challenge's duplicate rule; 20-500 Hz band-pass.",
                            "CodeURL": "https://github.com/JeetuSK0808/munitquest", "License": "BSD-3-Clause", "Version": "0.1"}],
           "RuntimeEnvironment": {"python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine()},
           "Execution": dict(meta, Runtime=float(max(meta.get("runtime_s", 0.0), 0.1)))}
    json.dump(log, open(out_dir / f"{rec}_desc-decomposed_log.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("task"); ap.add_argument("out", type=Path); ap.add_argument("--recs", default="")
    ap.add_argument("--iters", type=int, default=80); ap.add_argument("--ext", type=int, default=12); ap.add_argument("--seeds", default="1909")
    ap.add_argument("--threads", type=int, default=2); ap.add_argument("--min-sil", type=float, default=0.0)
    ap.add_argument("--refine", action="store_true", help="refine every candidate in the scorer's space before merging")
    ap.add_argument("--cands", type=Path, default=None, help="cache dir for raw CBSS candidates (reused if present)")
    ap.add_argument("--peel-sil", type=float, default=0.9, help="peel off sources above this SIL (muniverse default 0.9)")
    ap.add_argument("--peel-cov", type=float, default=0.35, help="peel off sources below this CoV-ISI (muniverse default 0.35)")
    ap.add_argument("--cbss", default="", help="extra FastIcaCBSS parameters, e.g. ica_opt_fun_exp=2,ica_initalization=activity_idx")
    a = ap.parse_args()
    num = lambda v: int(v) if v.lstrip("-").isdigit() else (float(v) if v.replace(".", "", 1).replace("e-", "", 1).lstrip("-").isdigit() else v)
    extra = {k: num(v) for k, v in (kv.split("=") for kv in a.cbss.split(",") if kv)}
    from muniverse.algorithms.cbss import FastIcaCBSS
    from run_scd import read_recording
    recs = sorted(p for p in (ROOT / "data" / a.task).iterdir() if p.is_dir() and p.name.startswith("recording-"))
    if a.recs:
        keep = {f"recording-{r.zfill(3)}" for r in a.recs.split(",")}; recs = [p for p in recs if p.name.split("_")[0] in keep]
    a.out.mkdir(parents=True, exist_ok=True)
    for rec in recs:
        if (a.out / f"{rec.name}_desc-decomposed_events.tsv").exists():
            print("skip", rec.name, flush=True); continue
        t0 = time.time(); X, fs, labels = read_recording(rec); Y = bandpass(X, fs).T
        units = []; per_seed = []
        peel_tag = "" if (a.peel_sil, a.peel_cov) == (0.9, 0.35) else f"_peel{a.peel_sil:g}-{a.peel_cov:g}"
        peel_tag += "".join(f"_{k}{v}" for k, v in extra.items())
        cache = (a.cands / f"{rec.name}_s{a.seeds.replace(',', '-')}_it{a.iters}_ext{a.ext}{peel_tag}.npz") if a.cands else None
        if cache is not None and cache.exists():
            z = np.load(cache, allow_pickle=True); units = list(zip(z["q"].tolist(), list(z["spikes"]))); per_seed = z["per_seed"].tolist()
        for seed in ([] if units else [int(s) for s in a.seeds.split(",")]):
            m = FastIcaCBSS(); m.set_parameter(random_seed=seed, ica_iterations=a.iters, ext_fact=a.ext,
                                               peel_off_sil_th=a.peel_sil, peel_off_cov_th=a.peel_cov, verbose=False, **extra)
            spikes, sources, scores = m.fit_predict(sig=Y, fsamp=fs)
            sil = np.asarray(scores["sil"], float); cov = np.asarray(scores["cov_isi"], float)
            byu = spikes_by_unit(spikes); n0 = len(units)
            for u, s in byu.items():
                k = u                                                     # unit_id = CBSS iteration index (0-based) = index into scores
                q = float(sil[k]) if 0 <= k < len(sil) else 0.0
                c = float(cov[k]) if 0 <= k < len(cov) else np.inf
                if q >= a.min_sil:
                    units.append((q - 0.1 * max(0.0, min(c, 2.0) - 0.3), s))     # quality: SIL, lightly penalised for irregular firing
            per_seed.append(dict(seed=seed, units=len(units) - n0))
        if cache is not None and not cache.exists():
            cache.parent.mkdir(parents=True, exist_ok=True)
            sp = np.empty(len(units), dtype=object); sp[:] = [np.asarray(u[1], np.int64) for u in units]
            np.savez(cache, q=np.array([u[0] for u in units]), spikes=sp, per_seed=np.array(per_seed, dtype=object))
        n_cand = len(units)
        if a.refine:
            units = refine_all(Y, units, fs)
        kept = merge_units(units, fs)
        meta = dict(recording=rec.name, fs=fs, channels=len(labels), seeds=a.seeds, ica_iterations=a.iters, ext_fact=a.ext,
                    peel_off_sil_th=a.peel_sil, peel_off_cov_th=a.peel_cov, **extra,
                    candidates=n_cand, after_refine=len(units) if a.refine else None, kept=len(kept), runtime_s=round(time.time() - t0, 1), per_seed=per_seed)
        write(a.out, rec.name, kept, fs, meta)
        print(f"{rec.name}: {n_cand} candidates -> {len(units)} after refine -> {len(kept)} units, {meta['runtime_s']} s", flush=True)


if __name__ == "__main__":
    main()
