# core/pipeline.py
import sys, subprocess, time, pathlib

def run_step(script, *args):
    cmd = [sys.executable, script, *args]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

if __name__=="__main__":
    if len(sys.argv)<2:
        print("Usage: python core/pipeline.py <audio_file>")
        sys.exit(1)
    audio = sys.argv[1]
    t0 = time.time()
    # Ensure outputs folder exists
    pathlib.Path("outputs").mkdir(exist_ok=True)
    # Transcribe
    run_step("core/transcribe.py", audio)
    # Summarise
    run_step("core/summarise.py")
    # Write metrics
    latency = time.time() - t0
    import json
    json.dump({"latency_s": round(latency,1)}, open("outputs/metrics.json","w"))
    print(f"Pipeline complete in {latency:.1f}s")
