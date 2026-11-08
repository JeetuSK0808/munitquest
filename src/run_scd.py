"""Decompose MUnitQuest HD-sEMG recordings with Swarm-Contrastive Decomposition (SCD, BSD-3, github.com/AgneGris/swarm-contrastive-decomposition)
and write the competition's BIDS events files.

usage: python src/run_scd.py <task: isometric|dynamic> <out_dir> [--recs 001,002] [--max-iter 150] [--ext auto|R] [--sil 0.8] [--threads 4]

Output per recording: <out_dir>/<recording>_desc-decomposed_events.tsv (onset, duration, sample, unit_id, event_type) plus
<out_dir>/<recording>_scd.json (runtime, config, per-unit silhouettes)."""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read_recording(rec_dir: Path):
    """EMG channels (type EMG in channels.tab/.tsv, status not bad) as float32 (time, channels), plus the sampling rate."""
    import pandas as pd
    from pyedflib.highlevel import read_edf
    name = rec_dir.name
    ch_file = rec_dir / f"{name}_channels.tab"
    if not ch_file.exists():  # Showdown data ships BIDS-standard _channels.tsv
        ch_file = rec_dir / f"{name}_channels.tsv"
    ch = pd.read_csv(ch_file, sep="\t")
    if "status" in ch.columns:  # drop channels the organisers flagged bad
        ch = ch[ch["status"].astype(str).str.lower() != "bad"]
    signals, sig_headers, _ = read_edf(str(rec_dir / f"{name}_emg.edf"))
    labels = [h["label"] for h in sig_headers]
    emg = [i for i, l in enumerate(labels) if l in set(ch.loc[ch["type"].str.upper() == "EMG", "name"])]
    fs = float(sig_headers[emg[0]]["sample_frequency"] if "sample_frequency" in sig_headers[emg[0]] else sig_headers[emg[0]]["sample_rate"])
    X = np.stack([np.asarray(signals[i], dtype=np.float32) for i in emg], axis=1)
    return X, fs, [labels[i] for i in emg]


def write_events(path: Path, timestamps, fs: float):
    rows = []
    for u, ts in enumerate(timestamps):
        ts = np.unique(np.asarray(ts.cpu().numpy() if hasattr(ts, "cpu") else ts, dtype=np.int64))
        rows += [(s / fs, 0, int(s), u, "motor-unit-spike") for s in ts]
    rows.sort(key=lambda r: (r[2], r[3]))
    with open(path, "w") as f:
        f.write("onset\tduration\tsample\tunit_id\tevent_type\n")
        for o, d, s, u, e in rows:
            f.write(f"{o:.6f}\t{d}\t{s}\t{u}\t{e}\n")
    return len(timestamps), len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["isometric", "dynamic"]); ap.add_argument("out", type=Path)
    ap.add_argument("--recs", default=""); ap.add_argument("--max-iter", type=int, default=150)
    ap.add_argument("--ext", default="auto"); ap.add_argument("--sil", type=float, default=0.8)
    ap.add_argument("--threads", type=int, default=4); ap.add_argument("--config", default="surface"); ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.threads)
    import scd
    from scd.train import preprocess_data, train_model
    data_root = ROOT / "data" / a.task
    recs = sorted(p for p in data_root.iterdir() if p.is_dir() and p.name.startswith("recording-"))
    if a.recs:
        keep = {f"recording-{r.zfill(3)}" for r in a.recs.split(",")}
        recs = [p for p in recs if p.name.split("_")[0] in keep]
    a.out.mkdir(parents=True, exist_ok=True)
    for rec in recs:
        out_tsv = a.out / f"{rec.name}_desc-decomposed_events.tsv"
        if out_tsv.exists():
            print("skip", rec.name, flush=True); continue
        t0 = time.time(); X, fs, labels = read_recording(rec)
        cfg = scd.load_config(a.config)
        over = dict(sampling_frequency=fs, bad_channels=[], max_iterations=a.max_iter, acceptance_silhouette=a.sil,
                    save_data=False, output_final_source_plot=False, output_source_plot=False, verbose_mode=False, device=a.device,
                    extension_factor=None if a.ext == "auto" else int(a.ext))
        for k, v in over.items():
            if hasattr(cfg, k): setattr(cfg, k, v)
        Xt = preprocess_data(torch.from_numpy(X), cfg)
        dictionary, ts = train_model(Xt, cfg)
        n_units, n_spikes = write_events(out_tsv, ts, fs)
        sil = dictionary.get("silhouettes") if isinstance(dictionary, dict) else None
        sil = [float(s) for s in (sil.tolist() if hasattr(sil, "tolist") else (sil or []))]
        meta = dict(recording=rec.name, fs=fs, channels=len(labels), samples=int(X.shape[0]), units=n_units, spikes=n_spikes,
                    runtime_s=round(time.time() - t0, 1), extension_factor=getattr(cfg, "extension_factor", None),
                    max_iterations=a.max_iter, acceptance_silhouette=a.sil, config=a.config, silhouettes=sil)
        json.dump(meta, open(a.out / f"{rec.name}_scd.json", "w"), indent=1)
        print(f"{rec.name}: {n_units} units, {n_spikes} spikes, {meta['runtime_s']} s, R={meta['extension_factor']}", flush=True)


if __name__ == "__main__":
    main()
