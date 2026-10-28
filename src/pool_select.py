"""Pool decomposed units from several runs and keep the set the official scorer rewards most.

Per recording: gather every unit from the given run dirs (optionally also raw CBSS candidate caches, plus their scorer-space
refined trains), compute each unit's official label-free metrics exactly as the scorer does but without its duplicate step
(PostProcessCBSS fit_from_spikes -> SIL / PNR, CoD-ISI -> binned metric_score, 0 if SIL < 0.9 or < 10 spikes), build the
same-motor-unit graph (the scorer's matching: lag = peak of the binned cross-correlation, common spikes within 1 ms at that
lag; but over +-50 ms instead of +-10 ms, and the shared fraction is taken relative to the SHORTER train, >= 0.3 = same unit. The scorer divides by the
longer train, which would let a partial copy of a unit count as a second one; we never write such copies), then choose the set with the largest
total label-free score (exact maximum-weight independent set, scipy milp). Units are written best-first. With --tail N, up to N
remaining units that duplicate nothing already written are appended (ordered by SIL): label-free they score 0, but the scorer
never penalises an extra unit and a unit that matches a ground-truth unit (F1 >= 0.5) earns its F1 (dynamic task = labels only).
The scorer drops a unit that duplicates ANY earlier unit in the file, so appended units cannot affect the ones before them.

usage: python src/pool_select.py <task> <out_dir> --runs runs/a,runs/b [--cands cache/cands --cand-glob '*it150*'] [--recs 060]
       [--refine] [--pre-sil 0.85] [--tail 60 --tail-sil 0.5]"""
import argparse, json, platform, sys, tempfile, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from proxy_score import bandpassed_edf, MUnitQuestScoring, ScoringConfig, EMPTY_GT  # noqa: E402

SAME_MU_LAG = 0.05   # s, lag range of the same-motor-unit test
METRIC_STEPS = [{"step": "validate_prediction"}, {"step": "fit_from_spikes"}, {"step": "get_discharge_metric", "metric": "cod_isi"}]


def write_tsv(path, trains, fs):
    rows = [(int(s), u) for u, S in enumerate(trains) for s in S]
    rows.sort()
    with open(path, "w") as f:
        f.write("onset\tduration\tsample\tunit_id\tevent_type\n")
        for s, u in rows:
            f.write(f"{s / fs:.6f}\t0\t{s}\t{u}\tmotor-unit-spike\n")


def unit_metrics(trains, edf, fs):
    """Official per-unit label-free metrics for every train (no duplicate removal)."""
    if not EMPTY_GT.exists():
        EMPTY_GT.parent.mkdir(parents=True, exist_ok=True); EMPTY_GT.write_text("onset\tduration\tsample\tunit_id\tevent_type\n")
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "all_events.tsv"; write_tsv(p, trains, fs)
        s = MUnitQuestScoring(prediction=str(p), cfg=ScoringConfig(fsamp=fs, steps=METRIC_STEPS, data=str(edf),
                                                                   ground_truth=str(EMPTY_GT), signal_metrics=True))
        s.get_score()
        df = s.unit_metrics.copy()
    df = df.sort_values("unit_id").reset_index(drop=True)
    df.loc[df["status"] != "good", "metric_score"] = 0.0
    return df


def lag_counts(a, b, ms):
    """Exact coincidence counts of a vs b + d for d in [-ms, ms] (a, b sorted unique ints; the binned cross-correlation)."""
    lo = np.searchsorted(a, b - ms); hi = np.searchsorted(a, b + ms, side="right"); cnt = np.zeros(2 * ms + 1, int)
    for k in range(int((hi - lo).max()) if len(b) else 0):
        m = lo + k < hi
        d = a[(lo + k)[m]] - b[m]
        cnt += np.bincount(d + ms, minlength=2 * ms + 1)
    return cnt


def shared_fraction(a, b, ms, tol, thr=None, denom=min):
    """Scorer duplicate measure: lag maximising exact coincidences within +-ms samples, then common spikes within tol samples
    (greedy two-pointer, as muniverse match_spikes), divided by the SHORTER train (denom=min, our same-motor-unit rule) or by
    the longer one (denom=max, the scorer's own rule, which lets a subset of a unit's spikes pass as a new unit). With thr,
    returns early when the answer relative to thr is already certain."""
    if len(a) == 0 or len(b) == 0:
        return 0.0
    cnt = lag_counts(a, b, ms); lag = int(np.argmax(cnt)) - ms; L = denom(len(a), len(b))
    if thr is not None:
        if cnt.max() >= thr * L:
            return cnt.max() / L                                   # tp >= exact coincidences
        bb = b + lag; i = np.clip(np.searchsorted(bb, a), 1, len(bb) - 1)
        near = np.minimum(np.abs(bb[i] - a), np.abs(bb[i - 1] - a)) <= tol
        if near.sum() < thr * L:
            return near.sum() / L                                  # tp <= spikes of a with a partner within tol
    t2 = np.sort(b + lag); i = j = tp = 0
    while i < len(a) and j < len(t2):
        dt = a[i] - t2[j]
        if abs(dt) <= tol:
            tp += 1; i += 1; j += 1
        elif dt < -tol:
            i += 1
        else:
            j += 1
    return tp / L


def conflicts(trains, fs, thr=0.3):
    """Pairs of trains that are the same motor unit: >= thr of the shorter train's spikes shared (stricter than the scorer's
    rule, so it never sees a duplicate, and a partial copy of a unit is never counted as a second unit). Lags are searched over
    +-50 ms, not the scorer's +-10 ms: a copy of a unit shifted by 10-50 ms (a delayed filter of the same motor unit) is still
    the same unit, while distinct units stay well below 0.3 at any lag in that range."""
    ms = int(SAME_MU_LAG * fs); tol = int(np.floor(0.001 * fs + 1e-9))
    n = len(trains); E = []
    for i in range(n):
        for j in range(i + 1, n):
            if shared_fraction(trains[i], trains[j], ms, tol, thr) >= thr:
                E.append((i, j))
    return E


def mwis(w, E):
    """Exact maximum-weight independent set (scipy milp); w >= 0."""
    from scipy.optimize import milp, LinearConstraint, Bounds
    n = len(w)
    if n == 0:
        return []
    if not E:
        return [i for i in range(n) if w[i] > 0]
    A = np.zeros((len(E), n))
    for k, (i, j) in enumerate(E):
        A[k, i] = A[k, j] = 1
    r = milp(c=-np.asarray(w, float), constraints=LinearConstraint(A, -np.inf, 1), integrality=np.ones(n), bounds=Bounds(0, 1))
    x = np.round(r.x).astype(int) if r.x is not None else np.zeros(n, int)
    return [i for i in range(n) if x[i] == 1 and w[i] > 0]


def load_run_units(run_dir, rec):
    p = Path(run_dir) / f"{rec}_desc-decomposed_events.tsv"
    if not p.exists():
        return []
    df = pd.read_table(p)
    return [np.unique(g["sample"].to_numpy(np.int64)) for _, g in df.groupby("unit_id")]


def fast_sil(X, S, fs):
    """The scorer's SIL at zero delay (its fit also tries every shift within +-10 ms and keeps the best, so it is >= this)."""
    from muniverse.evaluation.evaluate import pseudo_sil_score
    S = S[(S >= 0) & (S < X.shape[1])]
    if len(S) < 10:
        return -1.0
    w = X[:, S].mean(1); w /= np.linalg.norm(w) + 1e-12
    r = pseudo_sil_score(source=w @ X, spikes=S, fsamp=fs, min_peak_dist=0.01, match_dist=0.001)
    return float(r[0] if isinstance(r, tuple) else r)


def select(trains, fsil, edf, fs, pre_sil=0.85, tail=0, tail_sil=0.5):
    """Exact max-weight non-duplicate set over the units with zero-delay SIL >= pre_sil (weights = official label-free
    metric_score), written best-first; then up to `tail` more units (zero-delay SIL >= tail_sil, best first) that duplicate
    nothing already written. Returns (order of unit indices, n_scored, label-free total, per-unit table)."""
    n = len(trains)
    cand = [i for i in range(n) if len(trains[i]) >= 10 and fsil[i] >= pre_sil]
    tab = pd.DataFrame({"n_spikes": [len(S) for S in trains], "sil0": fsil})
    for c in ("sil", "pnr", "cod_isi", "metric_score"):
        tab[c] = np.nan
    order, tot = [], 0.0
    if cand:
        m = unit_metrics([trains[i] for i in cand], edf, fs)
        for c in ("sil", "pnr", "cod_isi", "metric_score"):
            tab.loc[cand, c] = m[c].to_numpy()
        w = m["metric_score"].to_numpy(float)
        sel = mwis(w, conflicts([trains[i] for i in cand], fs))
        sel.sort(key=lambda i: (-w[i], -m.loc[i, "sil"]))
        order = [cand[i] for i in sel]; tot = float(w[sel].sum())
    n_scored = len(order)
    if tail:
        ms, tol = int(SAME_MU_LAG * fs), int(np.floor(0.001 * fs + 1e-9)); chosen = set(order)
        for i in sorted((i for i in range(n) if i not in chosen and len(trains[i]) >= 10 and fsil[i] >= tail_sil), key=lambda i: -fsil[i]):
            if len(order) - n_scored >= tail:
                break
            if all(shared_fraction(trains[i], trains[j], ms, tol, 0.3) < 0.3 for j in order):
                order.append(i)
    tab["rank"] = -1
    tab.loc[order, "rank"] = np.arange(len(order))
    return order, n_scored, tot, tab


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("task"); ap.add_argument("out", type=Path)
    ap.add_argument("--runs", default=""); ap.add_argument("--recs", default="")
    ap.add_argument("--cands", type=Path, default=None, help="raw CBSS candidate cache dir (run_cbss --cands)")
    ap.add_argument("--cand-glob", default="*", help="which cache files to pool for a recording (comma-separated globs after the recording name)")
    ap.add_argument("--refine", action="store_true", help="also add the scorer-space refined train of every pooled unit")
    ap.add_argument("--pre-sil", type=float, default=0.85, help="official metrics only for units with zero-delay SIL >= this")
    ap.add_argument("--tail", type=int, default=0, help="append up to this many further non-duplicate units (label-based upside)")
    ap.add_argument("--tail-sil", type=float, default=0.5, help="minimum zero-delay SIL of a tail unit")
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    runs = [r for r in a.runs.split(",") if r]
    recs = sorted(p for p in (ROOT / "data" / a.task).iterdir() if p.is_dir() and p.name.startswith("recording-"))
    if a.recs:
        want = {f"recording-{r.zfill(3)}" for r in a.recs.split(",")}; recs = [p for p in recs if p.name.split("_")[0] in want]
    cache = ROOT / "cache" / a.task; cache.mkdir(parents=True, exist_ok=True)
    for rec_dir in recs:
        rec = rec_dir.name; t0 = time.time()
        fs = float(json.load(open(rec_dir / f"{rec}_emg.json"))["SamplingFrequency"])
        edf = bandpassed_edf(rec_dir, cache)
        trains, src, upstream_s = [], [], 0.0
        for r in runs:
            u = load_run_units(r, rec); trains += u; src += [Path(r).name] * len(u)
            lp = Path(r) / f"{rec}_desc-decomposed_log.json"
            if lp.exists():
                upstream_s += float((json.load(open(lp)).get("Execution") or {}).get("runtime_s", 0.0))
        if a.cands is not None:
            for f in sorted({f for g in a.cand_glob.split(",") for f in a.cands.glob(f"{rec}_{g}.npz")}):
                z = np.load(f, allow_pickle=True); u = [np.unique(np.asarray(s, np.int64)) for s in z["spikes"]]
                upstream_s += float(z["runtime_s"]) if "runtime_s" in z.files else 0.0
                trains += u; src += [f.stem] * len(u)
        if not trains:
            print(rec, "no units", flush=True); continue
        from pyedflib.highlevel import read_edf
        from muniverse.algorithms.post_processing import PostProcessCBSS
        from refine_units import refine
        sig, _, _ = read_edf(str(edf)); pp = PostProcessCBSS(steps=[], verbose=False)
        X = pp._whitening(pp._extension(np.asarray(sig, np.float64))); del sig
        seen = {}; keep = []                                              # drop exact duplicate trains
        for k, S in enumerate(trains):
            if len(S) >= 10 and S.tobytes() not in seen:
                seen[S.tobytes()] = k; keep.append(k)
        trains = [trains[k] for k in keep]; src = [src[k] for k in keep]; fsil, extra = [], []
        for k in range(len(trains)):
            if a.refine:                                                   # refined train added as an extra candidate
                S, s0, s1 = refine(X, trains[k], fs, 8); fsil.append(s0 if np.isfinite(s0) else -1.0)
                if np.isfinite(s1) and len(S) >= 10 and S.tobytes() not in seen and s0 >= 0.5:
                    seen[S.tobytes()] = -1; extra.append((S, src[k] + "+ref", s1))
            else:
                fsil.append(fast_sil(X, trains[k], fs))
        for S, sname, s1 in extra:
            trains.append(S); src.append(sname); fsil.append(s1)
        del X
        fsil = np.asarray(fsil, float)
        order, n_scored, tot, tab = select(trains, fsil, edf, fs, pre_sil=a.pre_sil, tail=a.tail, tail_sil=a.tail_sil)
        chosen = [trains[i] for i in order]
        tab.insert(0, "source", src); tab.to_csv(a.out / f"{rec}_pool_units.csv", index=False)
        write_tsv(a.out / f"{rec}_desc-decomposed_events.tsv", chosen, fs)
        pooled_from = runs + ([f"{a.cands}/{rec}_{a.cand_glob}.npz"] if a.cands else [])
        log = {"GeneratedBy": [{"Name": "BHopper-MUniverseCBSS-ensemble",
                               "Description": "MUniverse FastIcaCBSS decompositions (several seeds / peel-off settings, 20-500 Hz band-pass), "
                                              "scorer-space spike-triggered refinement, then the non-duplicate set (challenge duplicate rule) "
                                              "with the best label-free quality, best first.",
                               "CodeURL": "https://github.com/JeetuSK0808/munitquest", "License": "BSD-3-Clause", "Version": "0.2"}],
               "RuntimeEnvironment": {"python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine()},
               "Execution": dict(recording=rec, fs=fs, pooled_from=pooled_from, refine=a.refine, pre_sil=a.pre_sil, tail=a.tail,
                                 pooled_units=len(trains), scored_units=n_scored, written_units=len(chosen),
                                 runtime_s=round(time.time() - t0, 1),
                                 Runtime=float(round(time.time() - t0 + upstream_s, 1)))}   # s, all stages that fed this file
        json.dump(log, open(a.out / f"{rec}_desc-decomposed_log.json", "w"), indent=1)
        print(f"{rec}: pooled {len(trains)} -> {n_scored} selected (label-free sum {tot:.2f}), {len(chosen)} written, "
              f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
