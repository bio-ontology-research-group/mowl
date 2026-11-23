#!/usr/bin/env bash
# Run the lines of a benchmark manifest (see scripts/benchmark_manifests.py), PAR at a time.
#
#   MANIFEST=m.tsv RESULTS=dir [DEVICE=cpu] [PAR=4] [JVM_MEM=10g] [PYTHON=python] \
#       [NSHARDS=1 SHARD=1] scripts/run_manifest.sh
#
# Manifest line: tag <TAB> benchmark <TAB> method <TAB> epochs <TAB> seed <TAB> extra arguments.
# Each line runs
#   $PYTHON -m mowl.benchmarks run BENCHMARK --method METHOD --epochs EPOCHS --seed SEED \
#       --device $DEVICE --jvm-memory $JVM_MEM --workdir $RESULTS/work/TAG --out $RESULTS/TAG.json EXTRA
# with its log in $RESULTS/logs/TAG.log. A line whose result exists is skipped; a failed line
# leaves no result (its log says why). With NSHARDS > 1 only the lines whose
# (line number - 1) % NSHARDS == SHARD - 1 are run (a Slurm array task: SHARD=$SLURM_ARRAY_TASK_ID).
set -uo pipefail
MANIFEST=${MANIFEST:?set MANIFEST}; RESULTS=${RESULTS:?set RESULTS}
DEVICE=${DEVICE:-cpu}; PAR=${PAR:-4}; JVM_MEM=${JVM_MEM:-10g}; PYTHON=${PYTHON:-python}
NSHARDS=${NSHARDS:-1}; SHARD=${SHARD:-1}
mkdir -p "$RESULTS/logs" "$RESULTS/work"
run_one() {
  local TAG=$1 BENCHMARK=$2 METHOD=$3 EPOCHS=$4 SEED=$5 EXTRA=$6
  local OUT=$RESULTS/$TAG.json
  if [[ -s "$OUT" ]]; then echo "exists: $TAG"; return; fi
  mkdir -p "$RESULTS/work/$TAG"
  local t0=$SECONDS
  # shellcheck disable=SC2086
  if $PYTHON -m mowl.benchmarks run "$BENCHMARK" --method "$METHOD" --epochs "$EPOCHS" \
      --seed "$SEED" --device "$DEVICE" --jvm-memory "$JVM_MEM" --workdir "$RESULTS/work/$TAG" \
      --out "$OUT.tmp" $EXTRA > "$RESULTS/logs/$TAG.log" 2>&1 < /dev/null; then
    mv "$OUT.tmp" "$OUT"; echo "done: $TAG ($((SECONDS - t0)) s)"
  else
    rm -f "$OUT.tmp"; echo "FAILED: $TAG ($((SECONDS - t0)) s): $(tail -n 1 "$RESULTS/logs/$TAG.log")"
  fi
}
n=0
while IFS=$'\t' read -r TAG BENCHMARK METHOD EPOCHS SEED EXTRA; do
  [[ -z "$TAG" || "$TAG" == \#* ]] && continue
  n=$((n+1))
  (( (n-1) % NSHARDS == SHARD - 1 )) || continue
  while (( $(jobs -rp | wc -l) >= PAR )); do wait -n; done
  run_one "$TAG" "$BENCHMARK" "$METHOD" "$EPOCHS" "$SEED" "$EXTRA" &
done < "$MANIFEST"
wait
