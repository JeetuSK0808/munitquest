#!/usr/bin/env bash
# usage: src/queue_sd_pool.sh <task_dir> <old_run> <out_run> <recs csv> <tag>
#   Showdown re-selection (scorer venv): pool the old final selection (first, so its units keep their label), both CBSS runs,
#   their raw candidates and the residual candidates, add refined + scoring-window-consistent variants, select by the Showdown
#   label-free metrics (exact replica), tail 40 -> runs/<out_run>. Log: logs/<out_run>_<tag>.log
cd "$(dirname "$0")/.."; task=$1; old=$2; out=$3; recs=$4; tag=$5
export VECLIB_MAXIMUM_THREADS=2 OMP_NUM_THREADS=2
for id in ${recs//,/ }; do
  until pmset -g batt | grep -q "AC Power"; do sleep 300; done
  rec=$(ls -d data/$task/recording-${id}_* | xargs -n1 basename)
  [ -f "runs/$out/${rec}_desc-decomposed_events.tsv" ] && continue
  caffeinate -dims nice -n 10 .venv_score/bin/python -B -W ignore src/pool_select.py "$task" "runs/$out" --recs "$id" \
    --runs "runs/$old,runs/${old}_p0,runs/${old}_p1" --cands "cache/cands_$task" \
    --cand-glob "s1909-7_it150_ext12*,resid-${old}_base_*" --refine --wrefine --showdown --pre-sil 0.6 --tail 40 2>&1 \
    | grep --line-buffered -E "pooled|no units|Error|Traceback|rror" >> logs/${out}_$tag.log
done
echo "$(date) DONE $tag" >> logs/${out}_$tag.log
