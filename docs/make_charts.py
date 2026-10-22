#!/usr/bin/env python3
"""Draws the README's charts as SVG, one light and one dark file each.

    python3 docs/make_charts.py

Colors are GitHub's Primer tokens, and the background is transparent, so the
charts sit on the page like the rest of the README. The numbers below are the
measurements the README reports; the charts carry no other data.
"""

import pathlib

OUT = pathlib.Path(__file__).resolve().parent / "charts"
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', Helvetica, Arial, sans-serif"
THEMES = {
    "light": dict(text="#1f2328", muted="#59636e", grid="#d1d9e0", gpu="#0969da", ane="#bc4c00", base="#818b98"),
    "dark": dict(text="#f0f6fc", muted="#9198a1", grid="#3d444d", gpu="#4493f8", ane="#db6d28", base="#656c76"),
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


def svg(width, height, body, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-label="{title}" font-family="{FONT}">\n'
            f'<title>{title}</title>\n{body}</svg>\n')


def text(x, y, value, fill, size=12, anchor="start", weight=400):
    return (f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" font-size="{size}" font-weight="{weight}" '
            f'text-anchor="{anchor}">{value}</text>\n')


def bar(x, y, width, height, fill):
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(width, 0):.1f}" height="{height}" rx="2" fill="{fill}"/>\n'


def rule(x1, y1, x2, y2, t):
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{t["grid"]}"/>\n'


def legend(x, y, t, items):
    out = ""
    for label, color in items:
        out += f'<rect x="{x}" y="{y - 9}" width="10" height="10" rx="2" fill="{color}"/>\n'
        out += text(x + 15, y, label, t["muted"])
        x += 34 + 7 * len(label)
    return out


def ttft_chart(t):
    width, panel = 720, 330
    body = legend(0, 14, t, [("Splash 1.3.0", t["base"]), ("This branch", t["gpu"])])
    for index, (label, stock, branch) in enumerate(TTFT):
        x0 = index * (panel + 40)
        scale = (panel - 120) / stock
        body += text(x0, 52, label, t["text"], 13, weight=600)
        body += rule(x0, 62, x0, 140, t)
        for row, (value, color) in enumerate(((stock, t["base"]), (branch, t["gpu"]))):
            y = 66 + row * 38
            body += bar(x0, y, value * scale, 32, color)
            note = f" ({stock / branch:.2f}x faster)" if row else ""
            body += text(x0 + value * scale + 8, y + 21, f"{value:.1f} s{note}", t["text"], 13)
    return svg(width, 146, body, "Time to first token, Splash 1.3.0 versus this branch")


def layer_chart(t):
    width, unit, x0 = 720, 13.8, 120
    body = legend(0, 14, t, [("GPU", t["gpu"]), ("Neural Engine", t["ane"])])
    for name, top in (("Splash 1.3.0", 34), ("This branch", 128)):
        branch = name == "This branch"
        body += text(0, top + 4, name, t["text"], 13, weight=600)
        body += text(0, top + 31, "GPU", t["muted"])
        body += text(0, top + 63, "Neural Engine", t["muted"])
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
    return svg(width, 206, body, "Schematic of GPU and Neural Engine work in one layer, before and after")


def throughput_chart(t):
    width, x0, row = 720, 260, 30
    scale = (width - x0 - 70) / 22
    body = legend(0, 14, t, [("Neural Engine", t["ane"]), ("GPU", t["gpu"])])
    for index, (label, value, device, *shown) in enumerate(THROUGHPUT):
        y = 30 + index * row
        body += text(x0 - 10, y + 16, label, t["muted"], 12, "end")
        body += bar(x0, y, value * scale, row - 8, t[device])
        body += text(x0 + value * scale + 8, y + 16, shown[0] if shown else f"{value:g}", t["text"], 12)
    body += rule(x0, 26, x0, 30 + len(THROUGHPUT) * row, t)
    return svg(width, 36 + len(THROUGHPUT) * row, body, "Matmul throughput by device and number format")


def decode_chart(t):
    width, height, left, right, top, bottom = 720, 206, 56, 24, 14, 168
    xmax, ylo, yhi = 30000, 80, 110

    def sx(value):
        return left + value / xmax * (width - left - right)

    def sy(value):
        return bottom - (value - ylo) / (yhi - ylo) * (bottom - top)

    body = ""
    for value in range(ylo, yhi + 1, 10):
        body += rule(left, sy(value), width - right, sy(value), t)
        body += text(left - 8, sy(value) + 4, f"{value} ms", t["muted"], 11, "end")
    for value in range(0, xmax + 1, 10000):
        label = f"{value // 1000}K context" if value == xmax else f"{value // 1000}K" if value else "0"
        body += text(sx(value), bottom + 18, label, t["muted"], 11, "end" if value == xmax else "middle")
    points = " ".join(f"{sx(context):.1f},{sy(ms):.1f}" for context, ms in DECODE)
    body += f'<polyline points="{points}" fill="none" stroke="{t["gpu"]}" stroke-width="2"/>\n'
    for context, ms in DECODE:
        body += f'<circle cx="{sx(context):.1f}" cy="{sy(ms):.1f}" r="4" fill="{t["gpu"]}"/>\n'
        body += text(sx(context), sy(ms) - 10, f"{ms} ms", t["text"], 12, "middle")
    return svg(width, height, body, "Decode step time versus context length")


def attention_chart(t):
    width, x0, row = 720, 90, 30
    scale = (width - x0 - 190) / 30
    body = ""
    for index, (label, stock, two) in enumerate(ATTENTION):
        y = 4 + index * row
        change = (two / stock - 1) * 100
        body += text(x0 - 10, y + 16, f"{label} context", t["muted"], 12, "end")
        body += bar(x0, y, change * scale, row - 8, t["ane"])
        body += text(x0 + change * scale + 8, y + 16, f"+{change:.0f}%  ({stock:.3f} to {two:.3f} ms)", t["text"], 12)
    body += rule(x0, 0, x0, 4 + len(ATTENTION) * row, t)
    return svg(width, 10 + len(ATTENTION) * row, body, "Two-page attention experiment, percent slower per context")


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
