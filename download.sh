#!/usr/bin/env bash
# MUnitQuest familiarization data (DaRUS, public) -> data/<task>/ ; official scorer + tutorials -> upstream/
set -uo pipefail; cd "$(dirname "$0")"
for pair in "isometric:10.18419/DARUS-6143" "dynamic:10.18419/DARUS-6144"; do
  t=${pair%%:*}; doi=${pair#*:}
  if [ ! -f data/$t.zip ]; then
    curl -sSL --retry 5 -o data/$t.zip.part "https://darus.uni-stuttgart.de/api/access/dataset/:persistentId/?persistentId=doi:$doi" && mv data/$t.zip.part data/$t.zip
  fi
  mkdir -p data/$t && unzip -q -o data/$t.zip -d data/$t && echo "OK $t $(du -sh data/$t | cut -f1)"
done
[ -d upstream/munitquest-algorithm-submission ] || git clone -q https://github.com/MUnitQuest/munitquest-algorithm-submission upstream/munitquest-algorithm-submission
[ -d upstream/MUnitQuest_tutorials ] || git clone -q https://github.com/MUnitQuest/MUnitQuest_tutorials upstream/MUnitQuest_tutorials
echo DONE
