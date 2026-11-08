#!/usr/bin/env bash
# Showdown run (v2, 25 Oct): per recording of data/<task_dir>, one at a time (resumable):
#   1. CBSS default peel-off  (2 seeds x 150 iterations, extension 12, refine)  -> runs/<run>_p0    (raw candidates cached)
#   2. CBSS loose peel-off    (same, peel-off CoV-ISI threshold 1.0)             -> runs/<run>_p1
#   3. pool_select p0 + p1 (strict same-motor-unit rule, exact best set)        -> runs/<run>_base
#   4. residual_pass: joint least-squares peel-off of the base units, CBSS (2 x 150) on the residual -> cached candidates
#   5. pool_select: p0 + p1 units + refined residual candidates, scored on the original recording, best set first, then up to
#      40 more distinct units as a tail (label-based upside only)              -> runs/<run>
# then packs submissions/<run>.zip (format-checked). Only on AC power (waits on battery), caffeinated, nice'd, 2 threads.
# usage: src/run_showdown.sh <task_dir> <run>        e.g. src/run_showdown.sh showdown_isometric sd_iso_v2
cd "$(dirname "$0")/.." || exit 1
task=$1; run=$2
mkdir -p logs runs cache/cands_$task submissions
log=logs/showdown_${run}.log
export VECLIB_MAXIMUM_THREADS=2 OMP_NUM_THREADS=2
py() { caffeinate -i -s nice -n 10 .venv/bin/python -B -W ignore "$@"; }
keep() { grep --line-buffered -E "candidates|pooled|peeled|skip|no units|Error|Traceback" >> "$log"; }
echo "$(date) START $task -> runs/$run" >> "$log"
for d in data/"$task"/recording-*/; do
  rec=$(basename "$d"); id=${rec#recording-}; id=${id%%_*}
  [ -f "runs/$run/${rec}_desc-decomposed_events.tsv" ] && continue
  until pmset -g batt | grep -q "AC Power"; do echo "$(date) on battery, waiting" >> "$log"; sleep 300; done
  py src/run_cbss.py "$task" "runs/${run}_p0" --recs "$id" --cands cache/cands_$task --iters 150 --seeds 1909,7 --refine 2>&1 | keep
  py src/run_cbss.py "$task" "runs/${run}_p1" --recs "$id" --cands cache/cands_$task --iters 150 --seeds 1909,7 --refine --peel-cov 1.0 2>&1 | keep
  py src/pool_select.py "$task" "runs/${run}_base" --recs "$id" --runs "runs/${run}_p0,runs/${run}_p1" 2>&1 | keep
  py src/residual_pass.py "$task" "runs/${run}_base" --recs "$id" --cands cache/cands_$task 2>&1 | keep
  py src/pool_select.py "$task" "runs/$run" --recs "$id" --runs "runs/${run}_p0,runs/${run}_p1" --cands cache/cands_$task \
    --cand-glob "resid-${run}_base_*" --refine --tail 40 2>&1 | keep
done
py src/pack_submission.py "runs/$run" "submissions/$run.zip" --data "data/$task" >> "$log" 2>&1
echo "$(date) SHOWDOWN DONE $run" >> "$log"
