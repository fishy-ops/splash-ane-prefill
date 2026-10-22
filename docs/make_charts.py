#!/usr/bin/env python3
"""Draws the README's charts as SVG, one light and one dark file each.

    python3 docs/make_charts.py

The numbers below are the measurements the README reports; the charts carry
no other data.
"""

import pathlib

OUT = pathlib.Path(__file__).resolve().parent / "charts"
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
THEMES = {
    "light": dict(surface="#fcfcfb", text="#0b0b0b", secondary="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7", gpu="#2a78d6", ane="#eb6834", base="#a9a79f"),
    "dark": dict(surface="#1a1a19", text="#f0efec", secondary="#c3c2b7", muted="#898781",
                 grid="#2c2c2a", axis="#383835", gpu="#3987e5", ane="#d95926", base="#6b6a65"),
}

# Time to first token of uncached prompts, median seconds: Splash 1.3.0, this branch.
TTFT = [("6,144-token prompt", 25.08, 14.65), ("30,000-token prompt", 143.3, 91.7)]
# Throughput of one large matmul, trillion operations per second.
THROUGHPUT = [
    ("ANE, int8 weights x int8 activations", 20.7, "ane"),
    ("ANE, fp16 weights", 12.7, "ane"),
    ("ANE, int8 weights x fp16 activations", 10.4, "ane"),
    ("GPU, 4-bit weights (prefill matmul)", 6.9, "gpu"),
    ("GPU, prefill attention", 6.4, "gpu"),
    ("ANE, prefill attention", 2.0, "ane", "1.7-2.2"),
]
# Decode: context tokens, GPU milliseconds per verify step.
DECODE = [(924, 88.6), (7780, 94.5), (26933, 102.2)]
# One attention layer, ms: context, stock kernel, two pages per step.
ATTENTION = [("2K", 0.135, 0.167), ("8K", 0.373, 0.418), ("32K", 1.148, 1.183), ("64K", 2.264, 2.472)]
# Schematic layer phases: label, short label, stock GPU width, branch GPU width,
# whether the branch's Neural Engine works through the phase ("gates": part of it).
LAYER = [("norm", "", 1, 1, False), ("input projections", "in", 14, 4.7, True),
         ("scan / attention", "scan / attn", 8, 8, "gates"), ("gate", "", 1, 1, False),
         ("output projections", "out", 8, 2.6, True), ("FFN", "FFN", 10, 10, True)]


def svg(width, height, t, body, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-label="{title}" font-family="{FONT}">\n'
            f'<title>{title}</title>\n<rect width="100%" height="100%" rx="8" fill="{t["surface"]}"/>\n'
            f'{body}</svg>\n')


def text(x, y, value, fill, size=12, anchor="start", weight=400):
    return (f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" font-size="{size}" font-weight="{weight}" '
            f'text-anchor="{anchor}">{value}</text>\n')


def bar(x, y, width, height, fill):
    """A horizontal bar square at its baseline and rounded at its data end."""
    r = min(4, width / 2, height / 2)
    return (f'<path d="M{x:.1f},{y:.1f} H{x + width - r:.1f} Q{x + width:.1f},{y:.1f} {x + width:.1f},{y + r:.1f} '
            f'V{y + height - r:.1f} Q{x + width:.1f},{y + height:.1f} {x + width - r:.1f},{y + height:.1f} '
            f'H{x:.1f} Z" fill="{fill}"/>\n')


def legend(x, y, t, items):
    out = ""
    for label, color in items:
        out += f'<rect x="{x}" y="{y - 9}" width="10" height="10" rx="2" fill="{color}"/>\n'
        out += text(x + 16, y, label, t["secondary"])
        x += 36 + 7 * len(label)
    return out


def ttft_chart(t):
    width, panel, top = 760, 340, 74
    body = text(24, 32, "Time to first token, uncached prompt (lower is better)", t["text"], 15, weight=600)
    body += legend(24, 56, t, [("Splash 1.3.0", t["base"]), ("This branch", t["gpu"])])
    for index, (label, stock, branch) in enumerate(TTFT):
        x0 = 24 + index * (panel + 32)
        scale = (panel - 110) / stock
        body += text(x0, top + 16, label, t["secondary"], 13, weight=600)
        body += f'<line x1="{x0}" y1="{top + 30}" x2="{x0}" y2="{top + 112}" stroke="{t["axis"]}"/>\n'
        for row, (value, color) in enumerate(((stock, t["base"]), (branch, t["gpu"]))):
            y = top + 34 + row * 40
            body += bar(x0, y, value * scale, 34, color)
            note = f"  ·  {stock / branch:.2f}x faster" if row else ""
            body += text(x0 + value * scale + 8, y + 22, f"{value:.1f} s{note}", t["text"], 13)
    return svg(width, 200, t, body, "Time to first token, Splash 1.3.0 versus this branch")


def layer_chart(t):
    width, unit, x0 = 760, 14.5, 130
    body = text(24, 32, "One layer during prompt processing (schematic, not to scale)", t["text"], 15, weight=600)
    body += legend(24, 56, t, [("GPU", t["gpu"]), ("Neural Engine", t["ane"])])
    for name, top in (("Splash 1.3.0", 78), ("This branch", 172)):
        branch = name == "This branch"
        body += text(24, top + 4, name, t["text"], 13, weight=600)
        body += text(24, top + 31, "GPU", t["secondary"])
        body += text(24, top + 63, "Neural Engine", t["secondary"])
        x = x0
        for label, short, stock, narrow, shared in LAYER:
            span = (narrow if branch else stock) * unit
            body += bar(x, top + 14, span - 2, 26, t["gpu"])
            if (shared is True and (branch or label == "FFN")) or (shared == "gates" and branch):
                body += bar(x, top + 46, span * (0.55 if shared == "gates" else 1) - 2, 26, t["ane"])
            fitting = next((name for name in (label, short) if name and 6.5 * len(name) + 12 < span), "")
            if fitting:
                body += text(x + span / 2, top + 31, fitting, "#ffffff", 11, "middle")
            x += span
    body += text(24, 268, "The branch gives the Neural Engine most of each projection, plus the gate blocks during "
                 "the scan and attention,", t["secondary"], 12)
    body += text(24, 286, "and sizes each split so both sides finish a phase together.", t["secondary"], 12)
    return svg(width, 304, t, body, "Schematic of GPU and Neural Engine work in one layer, before and after")


def throughput_chart(t):
    width, x0, row = 760, 290, 30
    scale = (width - x0 - 90) / 22
    body = text(24, 32, "Matmul throughput on the M4 Pro (trillion ops/s, higher is better)", t["text"], 15, weight=600)
    body += legend(24, 56, t, [("Neural Engine", t["ane"]), ("GPU", t["gpu"])])
    for index, (label, value, device, *shown) in enumerate(THROUGHPUT):
        y = 76 + index * row
        body += text(x0 - 10, y + 17, label, t["secondary"], 12, "end")
        body += bar(x0, y, value * scale, row - 6, t[device])
        body += text(x0 + value * scale + 8, y + 17, shown[0] if shown else f"{value:g}", t["text"], 12)
    body += f'<line x1="{x0}" y1="72" x2="{x0}" y2="{76 + len(THROUGHPUT) * row}" stroke="{t["axis"]}"/>\n'
    return svg(width, 86 + len(THROUGHPUT) * row, t, body, "Matmul throughput by device and number format")


def decode_chart(t):
    width, height, left, right, top, bottom = 760, 260, 70, 40, 56, 210
    xmax, ylo, yhi = 30000, 80, 110

    def sx(value):
        return left + value / xmax * (width - left - right)

    def sy(value):
        return bottom - (value - ylo) / (yhi - ylo) * (bottom - top)

    body = text(24, 32, "Decode: GPU time per verify step versus context length", t["text"], 15, weight=600)
    for value in range(ylo, yhi + 1, 10):
        body += f'<line x1="{left}" y1="{sy(value):.1f}" x2="{width - right}" y2="{sy(value):.1f}" stroke="{t["grid"]}"/>\n'
        body += text(left - 8, sy(value) + 4, f"{value} ms", t["muted"], 11, "end")
    for value in range(0, xmax + 1, 10000):
        body += text(sx(value), bottom + 18, f"{value // 1000}K" if value else "0", t["muted"], 11, "middle")
    body += text((left + width - right) / 2, bottom + 40, "context tokens", t["muted"], 11, "middle")
    points = " ".join(f"{sx(context):.1f},{sy(ms):.1f}" for context, ms in DECODE)
    body += f'<polyline points="{points}" fill="none" stroke="{t["gpu"]}" stroke-width="2"/>\n'
    for context, ms in DECODE:
        body += (f'<circle cx="{sx(context):.1f}" cy="{sy(ms):.1f}" r="5" fill="{t["gpu"]}" '
                 f'stroke="{t["surface"]}" stroke-width="2"/>\n')
        body += text(sx(context), sy(ms) - 12, f"{ms} ms", t["text"], 12, "middle")
    return svg(width, height, t, body, "Decode step time versus context length")


def attention_chart(t):
    width, x0, row = 760, 120, 32
    scale = (width - x0 - 200) / 30
    body = text(24, 32, "Experiment: two KV pages per attention step, change in time per layer", t["text"], 15,
                weight=600)
    body += text(24, 54, "Positive is slower than the stock kernel; the change was reverted.", t["secondary"], 12)
    for index, (label, stock, two) in enumerate(ATTENTION):
        y = 72 + index * row
        change = (two / stock - 1) * 100
        body += text(x0 - 10, y + 18, f"{label} context", t["secondary"], 12, "end")
        body += bar(x0, y, change * scale, row - 6, t["ane"])
        body += text(x0 + change * scale + 8, y + 18, f"+{change:.0f}%  ({stock:.3f} to {two:.3f} ms)", t["text"], 12)
    body += f'<line x1="{x0}" y1="68" x2="{x0}" y2="{72 + len(ATTENTION) * row}" stroke="{t["axis"]}"/>\n'
    return svg(width, 84 + len(ATTENTION) * row, t, body, "Two-page attention experiment, percent slower per context")


def main():
    OUT.mkdir(exist_ok=True)
    charts = {"ttft": ttft_chart, "layer": layer_chart, "throughput": throughput_chart,
              "decode": decode_chart, "attention-experiment": attention_chart}
    for name, draw in charts.items():
        for mode, theme in THEMES.items():
            (OUT / f"{name}-{mode}.svg").write_text(draw(theme))
    print(f"wrote {len(charts) * len(THEMES)} charts to {OUT}")


if __name__ == "__main__":
    main()
