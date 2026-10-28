"""Second CBSS pass on the residual EMG after a joint least-squares peel-off of every kept unit.

Per recording: band-pass the EMG as run_cbss does, take the units of a base run, and fit all their multichannel MUAP waveforms at
once by least squares (each unit's spike train convolved with a free waveform over -win..+win on every channel, so overlapping
discharges of different units are separated jointly and every lag is removed, unlike muniverse's per-source peel-off in the
whitened space). Subtract the fit, run FastIcaCBSS on the residual and cache its candidates in the run_cbss --cands format;
pool_select.py then refines and scores them on the ORIGINAL recording (the scorer never sees the residual).

usage: python src/residual_pass.py <task> <base_run> --recs 060 [--iters 150 --seeds 1909,7 --win 0.025 --cands cache/cands]"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from run_cbss import bandpass, spikes_by_unit  # noqa: E402
from pool_select import load_run_units  # noqa: E402


def joint_ls_residual(Y, units, W, ridge=1e-6):
    """Y (C, N). units: list of spike sample arrays. Returns Y minus the least-squares fit of sum_u conv(spikes_u, MUAP_u)."""
    from scipy import sparse
    C, N = Y.shape; nl = 2 * W + 1; rows, cols = [], []
    for u, S in enumerate(units):
        for k, l in enumerate(range(-W, W + 1)):
            r = S + l; ok = (r >= 0) & (r < N)
            rows.append(r[ok]); cols.append(np.full(ok.sum(), u * nl + k))
    rows = np.concatenate(rows); cols = np.concatenate(cols)
    D = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(N, len(units) * nl))
    G = (D.T @ D).toarray(); B = np.asarray(D.T @ Y.T)                      # (K, K), (K, C)
    G[np.diag_indices_from(G)] += ridge * np.trace(G) / len(G)
    T = np.linalg.solve(G, B)                                              # MUAP waveforms, (K, C)
    return Y - np.asarray(D @ T).T


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("task"); ap.add_argument("base", type=Path); ap.add_argument("--recs", required=True)
    ap.add_argument("--iters", type=int, default=150); ap.add_argument("--seeds", default="1909,7"); ap.add_argument("--ext", type=int, default=12)
    ap.add_argument("--win", type=float, default=0.025); ap.add_argument("--cands", type=Path, default=ROOT / "cache" / "cands")
    ap.add_argument("--peel-cov", type=float, default=0.35)
    a = ap.parse_args(); a.cands.mkdir(parents=True, exist_ok=True)
    from muniverse.algorithms.cbss import FastIcaCBSS
    from run_scd import read_recording
    want = {f"recording-{r.zfill(3)}" for r in a.recs.split(",")}
    recs = sorted(p for p in (ROOT / "data" / a.task).iterdir() if p.is_dir() and p.name.split("_")[0] in want)
    for rec in recs:
        tag = f"resid-{a.base.name}_s{a.seeds.replace(',', '-')}_it{a.iters}_ext{a.ext}" + ("" if a.peel_cov == 0.35 else f"_peelcov{a.peel_cov:g}")
        out = a.cands / f"{rec.name}_{tag}.npz"
        if out.exists():
            print("skip", rec.name, flush=True); continue
        t0 = time.time(); X, fs, labels = read_recording(rec); Y = bandpass(X, fs).T
        units = load_run_units(a.base, rec.name)
        R = joint_ls_residual(Y, units, int(round(a.win * fs)))
        ev = 1 - (R ** 2).sum() / (Y ** 2).sum()
        cands, per_seed = [], []
        for seed in [int(s) for s in a.seeds.split(",")]:
            m = FastIcaCBSS(); m.set_parameter(random_seed=seed, ica_iterations=a.iters, ext_fact=a.ext, peel_off_cov_th=a.peel_cov, verbose=False)
            spikes, _, scores = m.fit_predict(sig=R, fsamp=fs)
            sil = np.asarray(scores["sil"], float); n0 = len(cands)
            for u, s in spikes_by_unit(spikes).items():
                cands.append((float(sil[u]) if 0 <= u < len(sil) else 0.0, s))
            per_seed.append(dict(seed=seed, units=len(cands) - n0))
        sp = np.empty(len(cands), dtype=object); sp[:] = [np.asarray(c[1], np.int64) for c in cands]
        np.savez(out, q=np.array([c[0] for c in cands]), spikes=sp, per_seed=np.array(per_seed, dtype=object), runtime_s=time.time() - t0)
        print(f"{rec.name}: peeled {len(units)} units (explained {100 * ev:.1f}% of band-passed energy) -> {len(cands)} residual candidates, "
              f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
