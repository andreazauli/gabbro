#!/bin/bash
# community rows + concurrency check (one job that owns the GPU)
set -u
DEP=${DEP:?checkout with bin/}; M=${MODELS:?folder of .basalt packs}; R=${OUT:-$PWD/community}
INK=$M/jesdga--Inkling-2-Mini-Basalt/Inkling-2-Mini-NVFP4-MX8.basalt
Q27=$M/jesdga--Qwen3.8-27B-Basalt/Qwen3.8-27B-QUASAR-NVFP4.basalt
FN=$M/jesdga--Qwen3.8-Flash-Next-Basalt/Qwen3.8-FN-UD-IQ1_S.basalt
mkdir -p $R; cd $DEP
echo "== $(hostname) $(date -Is)"; free -g; bin/basalt-bench --hardware
for p in $INK $Q27 $FN; do
  n=$(basename $p .basalt); echo "=== community $n $(date -Is)"
  ( while sleep 5; do free -m | awk '/Mem:/{print $3}'; done ) > $R/$n.memtrace & MT=$!
  bin/basalt-bench --community --model $p --submitter gabbro --community-dir $R --logs $R/logs-$n > $R/$n.community.out 2> $R/$n.community.err
  echo "exit $?"; kill $MT
  cat $R/$n.community.out; tail -5 $R/$n.community.err
  echo "peak used MiB: $(sort -n $R/$n.memtrace | tail -1)"
done
PORT=18432
for p in $INK $FN; do
  n=$(basename $p .basalt); echo "=== parallel $n $(date -Is)"
  bin/basalt serve --model $p --host 127.0.0.1 --port $PORT --kv int8 --parallel 4 > $R/$n.parallel.serve.log 2>&1 & SP=$!
  for i in $(seq 1 300); do curl -s http://127.0.0.1:$PORT/health | grep -q '"loaded": *true' && break; kill -0 $SP 2>/dev/null || break; sleep 2; done
  .venv/bin/python "$(dirname "$0")"/conc.py $PORT 1 256
  .venv/bin/python "$(dirname "$0")"/conc.py $PORT 4 256
  free -g | head -2
  grep -E "slots|done" $R/$n.parallel.serve.log | tail -8
  kill $SP; wait $SP 2>/dev/null; sleep 3
done
echo "== end $(date -Is)"
