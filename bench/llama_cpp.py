# llama.cpp on the same benchmark as MLX-LM's and the Splash runs: per size, one
# warm-up, then 5 runs of a fresh prompt of exactly N tokens followed by 128
# generated tokens, timed by llama-server's own report.
#   python3 bench/llama_cpp.py MODEL.gguf   (llama-server on PATH)
import json, pathlib, statistics, subprocess, sys, sysconfig, time, urllib.request

GGUF, PORT = sys.argv[1], 8030
URL = f"http://127.0.0.1:{PORT}"
root = pathlib.Path(sysconfig.get_paths()["stdlib"])
TEXT = "\n".join(p.read_text(errors="replace") for p in sorted(root.glob("*.py")))


def post(path, body):
    request = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=3600) as response:
        return json.load(response)


def sized(start, tokens):
    chars = int(tokens * 3.6)
    for _ in range(8):
        prompt = TEXT[start:start + chars]
        n = len(post("/tokenize", {"content": prompt})["tokens"])
        if abs(n - tokens) <= 16:
            return prompt
        chars = int(chars * tokens / n)
    raise RuntimeError("could not size the prompt")


server = subprocess.Popen(["llama-server", "-m", GGUF, "--port", str(PORT), "-c", "32768", "-ngl", "99",
                           "--no-webui", "-np", "1"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    for _ in range(300):
        try:
            if json.load(urllib.request.urlopen(URL + "/health", timeout=2)).get("status") == "ok":
                break
        except OSError:
            time.sleep(1)
    print("llama.cpp server ready", flush=True)
    offset = 300000
    for tokens in (6144, 30000):
        runs = []
        for rep in range(6):
            prompt = sized(offset, tokens)
            offset += int(tokens * 3.6) + 4000
            r = post("/completion", {"prompt": prompt, "n_predict": 128, "ignore_eos": True,
                                     "cache_prompt": False, "temperature": 0})
            t = r["timings"]
            if rep == 0:
                print(f"  llama.cpp {tokens}: warm-up done", flush=True)
                continue
            runs.append((t["prompt_per_second"], t["predicted_per_second"]))
            print(f"  run {rep}: {t['prompt_n']} tok, TTFT {t['prompt_ms'] / 1000:.2f} s, prompt "
                  f"{t['prompt_per_second']:.1f} tok/s, generated {t['predicted_n']} at "
                  f"{t['predicted_per_second']:.1f} tok/s", flush=True)
        pp = [p for p, _ in runs]
        tg = [g for _, g in runs]
        print(f"llama.cpp {tokens}: prompt {statistics.median(pp):.1f} tok/s ({min(pp):.1f}-{max(pp):.1f}), "
              f"generation {statistics.median(tg):.1f} tok/s ({min(tg):.1f}-{max(tg):.1f})", flush=True)
finally:
    server.terminate()
    server.wait(timeout=60)
