# conc.py PORT N MAXTOK : N simultaneous chat completions; prints per-request and aggregate tok/s
import json, sys, threading, time, urllib.request
port, n, maxtok = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
topics = ["a lighthouse keeper", "a robot gardener", "a medieval cartographer", "a deep-sea diver",
          "a mountain village baker", "a lost satellite", "a jazz pianist", "an old library cat"]
res = [None] * n
def one(i):
    body = {"messages": [{"role": "user", "content": f"Write a long, detailed story about {topics[i % len(topics)]}. At least 600 words."}],
            "max_tokens": maxtok, "temperature": 0.7, "seed": i + 1,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=1200))
    res[i] = (time.time() - t0, r["usage"]["completion_tokens"], r.get("timings", {}).get("predicted_per_second"))
t0 = time.time()
th = [threading.Thread(target=one, args=(i,)) for i in range(n)]
[t.start() for t in th]; [t.join() for t in th]
wall = time.time() - t0
tot = sum(r[1] for r in res)
for i, r in enumerate(res): print(f"  req{i}: {r[1]} tokens in {r[0]:.1f}s  engine {r[2]} tok/s")
print(f"N={n} total_tokens={tot} wall={wall:.1f}s aggregate={tot/wall:.1f} tok/s")
