# MUnitQuest Algorithm Challenge: motor-unit decomposition (team BHopper)

Satyajeeth (Jeetu) Suresh Kannan, Innovation Academy, Alpharetta, Georgia, USA (Codabench user jeetu_sk).
Code for the MUnitQuest Algorithm Challenge (Codabench competition 15749; University of Stuttgart and Imperial College London):
decomposing high-density surface EMG into motor-unit spike trains, for the isometric and the dynamic task.
The challenge data is not included in this repository.

## Method (Showdown config v2)

Per recording (`src/run_showdown.sh <task_dir> <run>`):

1. **CBSS, default peel-off.** MUniverse `FastIcaCBSS` (2 seeds x 150 iterations, extension 12) on the 20-500 Hz band-passed EMG,
   then a spike-triggered refinement of every candidate in the scorer's own whitened space (`src/run_cbss.py`, `src/refine_units.py`).
2. **CBSS, loose peel-off.** The same, but every source with SIL > 0.9 is peeled off whatever its firing regularity.
3. **Strict pool** of 1 and 2 (`src/pool_select.py`). Every unit gets the challenge's label-free metrics (SIL, PNR, CoD-ISI,
   binned into the unit score; 0 below SIL 0.9 or 10 spikes), and the best set under the scorer's same-motor-unit rule is kept.
4. **Residual pass** (`src/residual_pass.py`). The multichannel MUAP waveforms of all kept units are fitted jointly by least squares
   and subtracted from the EMG, which removes every lag of every unit at once; CBSS then runs again on the residual.
5. **Final pool** of 1, 2 and the refined residual candidates, best first, plus up to 40 more distinct units.
6. **Packing** (`src/pack_submission.py`) with the organisers' validators and format checks.

## Local results (label-free proxy, not the official score)

`src/proxy_score.py` runs the label-free branch of the organisers' scorer on our band-passed copy of each recording, so every unit
is scored as unmatched. On 4 isometric recordings held out from all tuning, config v2 averages 19.12 per recording versus 17.60 for
a single default CBSS run (pooling +0.57, residual pass +0.95), with no recording worse.

## Showdown v3 re-selection (8 Nov 2026, dynamic task)

The Showdown scorer (MUnitQuest/munitquest-algorithm-submission 46bdda9, muniverse-emg 0.1.7) differs from the familiarization one:
only the window between `scoring_start` and `scoring_end` counts; unmatched units need SIL >= 0.85 and >= 25 in-window spikes;
PNR is binned from 28 dB; in the dynamic task each unmatched unit's score is multiplied by a "model recall". `src/showdown_score.py`
replicates its label-free branch (and has an exact vectorised per-unit version, checked against the organisers' code to 1e-15).

`src/queue_sd_pool.sh` re-selects every recording from the candidates already computed for v2 (both CBSS runs, their raw
candidates, the residual candidates), adding scorer-space refined and scoring-window-consistent refined variants
(`refine_units.wrefine`), and keeps the best set of distinct units by the Showdown label-free metrics (`src/pool_select.py
--showdown`). **Plausibility guard:** a unit the scorer would credit label-free is only written if its discharge pattern is
motor-unit-like (CoD-ISI < 0.372; a random Poisson-like train gives about 0.5) and its pulse stands clear of the noise (PNR >= 20 dB).
Without it, a big candidate pool fills up with noise sources that just clear the SIL cut (one 70 % MVC recording: 157 such
"units", PNR 13-15 dB, CoD about 0.5). Then up to 80 more distinct units that the scorer does not credit label-free (label-based
upside only).

Local result (label-free branch on our band-passed EDF, all 50 dynamic recordings): 11.06 -> 11.50 per recording; of the v2 file's
11.06, only 7.98 came from units that pass the guard, and v3 beats that on every recording. On a fixed 13-recording sample the
organisers' own code path gives 11.40 -> 13.70 (20-500 Hz EDF) and 11.05 -> 13.11 (10-500 Hz EDF, robustness check). Every v2 unit
that passes the guard with SIL >= 0.85 has the same motor unit in v3.

## Run

```bash
uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python -r requirements.txt
bash download.sh                                     # public familiarization data (DaRUS) + organisers' code -> upstream/
src/run_showdown.sh showdown_isometric sd_iso_v2     # reads data/showdown_isometric/recording-*/ (challenge BIDS layout)
```

The runner works one recording at a time (resumable), only on AC power, with 2 threads, and writes `submissions/<run>.zip`.

## Notes

- Built on MUniverse (`muniverse-emg` 0.1.4, https://github.com/dfarinagroup/muniverse) and the organisers' scoring and
  validation code (https://github.com/MUnitQuest/munitquest-algorithm-submission), which `download.sh` clones into `upstream/`.
- Written with AI assistance (Claude, Anthropic) under the author's direction.

## License

BSD-3-Clause (see `LICENSE`).
