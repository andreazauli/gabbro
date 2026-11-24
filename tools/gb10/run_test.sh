#!/bin/bash
# Usage (inside a job allocation that owns the GPU): run_test.sh DEPLOYDIR PACK OUTDIR [extra serve args...]
# Starts basalt serve on 127.0.0.1, one chat completion, stops it; then basalt-bench with and without drafting.
set -u
DEP=$1; PACK=$2; OUT=$3; shift 3
PORT=18431
mkdir -p "$OUT"
cd "$DEP"
export HOME=${HOME:-/tmp}
echo "== host $(hostname) $(date -Is)"; nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv
free -g

echo "== serve"
bin/basalt serve --model "$PACK" --host 127.0.0.1 --port $PORT --kv int8 "$@" > "$OUT/serve.log" 2>&1 &
SP=$!
ok=0
for i in $(seq 1 600); do
  if ! kill -0 $SP 2>/dev/null; then echo "serve exited"; break; fi
  if curl -s "http://127.0.0.1:$PORT/health" | grep -q '"loaded": *true'; then ok=1; break; fi
  sleep 2
done
curl -s "http://127.0.0.1:$PORT/health" > "$OUT/health.json"; echo; cat "$OUT/health.json" | head -c 600; echo
T0=$(date +%s.%N)
curl -s -m 600 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is the capital of Italy? One word."}],"max_tokens":256,"temperature":0}' > "$OUT/chat.json"
T1=$(date +%s.%N)
echo "chat took $(echo "$T1 - $T0" | bc) s"; cat "$OUT/chat.json"; echo
free -g; nvidia-smi --query-gpu=memory.used --format=csv
grep -E "done|ready|loaded|kv|slots|error|warn" "$OUT/serve.log" | tail -20
kill $SP; wait $SP 2>/dev/null
sleep 3

echo "== bench (default drafter)"
bin/basalt-bench --model "$PACK" --logs "$OUT/bench-spec" --seed 1 -t conv,struct,prose -p 16384 -- --kv int8 "$@" > "$OUT/bench-spec.md" 2> "$OUT/bench-spec.err"
cat "$OUT/bench-spec.md"; tail -5 "$OUT/bench-spec.err"
echo "== bench (no drafter)"
bin/basalt-bench --model "$PACK" --logs "$OUT/bench-nospec" --seed 1 --spec 0 -t conv,struct,prose -p 16384 -- --kv int8 "$@" > "$OUT/bench-nospec.md" 2> "$OUT/bench-nospec.err"
cat "$OUT/bench-nospec.md"; tail -5 "$OUT/bench-nospec.err"
echo "== end $(date -Is)"
