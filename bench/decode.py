#!/usr/bin/env python3
"""Decode speed after code contexts of increasing length.

    python3 bench/decode.py [--url URL] [--contexts 1024,8192,30000] [--tokens 512]

For each context, one greedy generation; reports the server's streaming rate
and, from /status counters across the request, the tokens each verify step
accepted and the GPU time per step. The two separate draft acceptance, which
depends on the text, from the step cost, which depends on the engine.
"""

import argparse
import json
import pathlib
import sysconfig
import urllib.request

ASK = "\n\nRefactor the last function above to remove duplication and add error handling. Show the full code."
CHARS_PER_TOKEN = 3.9


def call(url, body=None):
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(url, data, {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=3600) as response:
        return json.load(response)


def counters(url):
    """The numeric leaves of /status, by dotted path."""
    flat = {}

    def walk(node, prefix):
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, f"{prefix}{key}.")
        elif isinstance(node, (int, float)) and not isinstance(node, bool):
            flat[prefix[:-1]] = node

    walk(call(f"{url}/status"), "")
    return flat


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--contexts", default="1024,8192,30000")
    parser.add_argument("--tokens", type=int, default=512)
    args = parser.parse_args()

    model = call(f"{args.url}/v1/models")["data"][0]["id"]
    root = pathlib.Path(sysconfig.get_paths()["stdlib"])
    text = "\n".join(path.read_text(errors="replace") for path in sorted(root.glob("*.py")))
    for context in (int(value) for value in args.contexts.split(",")):
        prompt = text[-int(context * CHARS_PER_TOKEN):] + ASK
        before = counters(args.url)
        response = call(f"{args.url}/v1/chat/completions", {
            "model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": args.tokens, "temperature": 0, "reasoning_effort": "none",
        })
        after = counters(args.url)
        delta = {key: after[key] - before.get(key, 0) for key in after}
        steps = max(delta.get("scheduler.decode_batches", 0), 1)
        per_step = 1 + delta.get("metrics.accepted_draft_tokens", 0) / steps
        step_ms = delta.get("model_timing.decode.total_gpu_ms", 0) / steps
        rate = response["metrics"]["request_latency"].get("stream_tokens_per_second", 0)
        print(f"{response['usage']['prompt_tokens']:>7} context  {response['usage']['completion_tokens']:>5} tokens  "
              f"{rate:6.1f} tok/s  {per_step:4.2f} tokens/step  {step_ms:6.1f} ms/step", flush=True)


if __name__ == "__main__":
    main()
