#!/usr/bin/env python3
"""Uncached time to first token at a fixed prompt length.

    python3 bench/ttft.py [--url URL] [--tokens 6144] [--reps 3]

Measures a Splash server you started, e.g. `./splash serve --model ...` in the
engine checkout. Each prompt is a different slice of Python's standard library
source, sized to the target with the server's own count_tokens, so no request
starts from another's cached prefix. The first request after the server is
ready is the cold one.
"""

import argparse
import json
import pathlib
import statistics
import sysconfig
import urllib.request

SUFFIX = "\n\nSummarize the code above in one sentence."
CHARS_PER_TOKEN = 3.9


def call(url, body=None):
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(url, data, {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=3600) as response:
        return json.load(response)


def corpus():
    root = pathlib.Path(sysconfig.get_paths()["stdlib"])
    return "\n".join(path.read_text(errors="replace") for path in sorted(root.glob("*.py")))


def sized_prompt(url, model, text, start, tokens):
    """A prompt of `tokens` +-16 tokens starting at `start`, by the server's count."""
    chars = int(tokens * CHARS_PER_TOKEN)
    for _ in range(6):
        prompt = text[start:start + chars] + SUFFIX
        counted = call(f"{url}/v1/messages/count_tokens",
                       {"model": model, "messages": [{"role": "user", "content": prompt}]})["input_tokens"]
        if abs(counted - tokens) <= 16:
            return prompt
        chars = int(chars * tokens / counted)
    raise RuntimeError(f"could not size a prompt to {tokens} tokens")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--tokens", type=int, default=6144)
    parser.add_argument("--reps", type=int, default=3)
    args = parser.parse_args()

    model = call(f"{args.url}/v1/models")["data"][0]["id"]
    text = corpus()
    if len(text) < (args.reps + 1) * args.tokens * CHARS_PER_TOKEN:
        raise SystemExit("the standard library is too small for that many prompts of that length")
    times = []
    for rep in range(args.reps):
        prompt = sized_prompt(args.url, model, text, rep * len(text) // (args.reps + 1), args.tokens)
        response = call(f"{args.url}/v1/chat/completions", {
            "model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1, "temperature": 0, "reasoning_effort": "none",
        })
        cached = response["metrics"]["cache"]["matched_tokens"]
        seconds = response["metrics"]["request_latency"]["start_to_first_token_ms"] / 1000
        times.append(seconds)
        print(f"{response['usage']['prompt_tokens']:>7} tokens  {cached:>5} cached  {seconds:7.2f} s", flush=True)
    print(f"median {statistics.median(times):.2f} s, first {times[0]:.2f} s")


if __name__ == "__main__":
    main()
