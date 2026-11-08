"""Label-free local proxy of the MUnitQuest score: the official scorer's signal-based branch (duplicate removal, CBSS refit from the
predicted spikes, SIL / PNR / CoD-ISI confidence) with no ground truth, so every predicted unit is scored as "unmatched".
Uses the organisers' own code (bundle/scoring/src/algorithm_scoring) on our 20-500 Hz band-passed copy of the recording
(the organisers score on their own preprocessed EDF, which is not public).

usage: python src/proxy_score.py <task> <pred_dir> [--recs 001,002]"""
import argparse, json, sys, tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "upstream/munitquest-algorithm-submission/bundle/scoring/src"))
from algorithm_scoring.score_recording import MUnitQuestScoring, ScoringConfig  # noqa: E402

STEPS = [
    {"step": "remove_duplicates", "max_shift": 0.01, "tolerance": 0.001, "threshold": 0.3, "mode": "first"},
    {"step": "validate_prediction"},
    {"step": "fit_from_spikes"},
    {"step": "get_discharge_metric", "metric": "cod_isi"},
]


def bandpassed_edf(rec_dir: Path, cache: Path, band=(20, 500)) -> Path:
    """EMG channels only, 4th-order zero-phase Butterworth band-pass (default 20-500 Hz), written once to an EDF cache."""
    out = cache / (f"{rec_dir.name}_bp.edf" if tuple(band) == (20, 500) else f"{rec_dir.name}_bp{band[0]}-{band[1]}.edf")
    if out.exists():
        return out
    from scipy.signal import butter, sosfiltfilt
    from pyedflib import highlevel
    sys.path.insert(0, str(ROOT / "src"))
    from run_scd import read_recording
    X, fs, labels = read_recording(rec_dir)
    sos = butter(4, list(band), btype="band", fs=fs, output="sos")
    Y = sosfiltfilt(sos, X.astype(np.float64), axis=0)
    heads = highlevel.make_signal_headers(labels, dimension="mV", sample_frequency=fs,
                                          physical_min=float(Y.min()) - 1e-6, physical_max=float(Y.max()) + 1e-6)
    highlevel.write_edf(str(out), [Y[:, i] for i in range(Y.shape[1])], heads)
    return out


EMPTY_GT = ROOT / "cache" / "empty_groundtruth_events.tsv"


def score_one(pred_tsv: Path, edf: Path, fs: float):
    if not EMPTY_GT.exists():
        EMPTY_GT.parent.mkdir(parents=True, exist_ok=True)
        EMPTY_GT.write_text("onset\tduration\tsample\tunit_id\tevent_type\n")
    cfg = ScoringConfig(fsamp=fs, steps=STEPS, data=str(edf), ground_truth=str(EMPTY_GT), signal_metrics=True)
    s = MUnitQuestScoring(prediction=str(pred_tsv), cfg=cfg)
    s.get_score()
    return s.metrics, s.unit_metrics


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("task"); ap.add_argument("pred", type=Path); ap.add_argument("--recs", default="")
    a = ap.parse_args()
    cache = ROOT / "cache" / a.task; cache.mkdir(parents=True, exist_ok=True)
    preds = sorted(a.pred.glob("*_desc-decomposed_events.tsv"))
    if a.recs:
        keep = {f"recording-{r.zfill(3)}" for r in a.recs.split(",")}
        preds = [p for p in preds if p.name.split("_")[0] in keep]
    rows = []
    for p in preds:
        rec = p.name.replace("_desc-decomposed_events.tsv", "")
        rec_dir = ROOT / "data" / a.task / rec
        fs = float(json.load(open(rec_dir / f"{rec}_emg.json"))["SamplingFrequency"])
        m, units = score_one(p, bandpassed_edf(rec_dir, cache), fs)
        units.to_csv(a.pred / f"{rec}_proxy_units.csv", index=False)
        rows.append(dict(recording=rec, **{k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v) for k, v in m.items()}))
        print(f"{rec}: proxy score {m['score']:.2f} | accepted {m.get('accepted_units')} rejected {m.get('rejected_units')} "
              f"| SIL {m.get('sil_mean', float('nan')):.3f} PNR {m.get('pnr_mean', float('nan')):.1f} CoD {m.get('cod_isi_mean', float('nan')):.3f}", flush=True)
    df = pd.DataFrame(rows); df.to_csv(a.pred / "proxy_scores.csv", index=False)
    if len(df):
        print(f"MEAN proxy score over {len(df)} recordings: {df['score'].mean():.2f}")


if __name__ == "__main__":
    main()
