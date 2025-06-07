#!/usr/bin/env python
import sys, subprocess, json, time, pathlib

audio = sys.argv[1]
t0 = time.time()
subprocess.run(["python", "core/transcribe.py", audio], check=True)
subprocess.run(["python", "core/summarise.py"], check=True)

metrics = {
    "latency_s": round(time.time() - t0, 1),
}
pathlib.Path("outputs").mkdir(exist_ok=True)
json.dump(metrics, open("outputs/metrics.json", "w"))
print(metrics)
