#!/usr/bin/env python3
"""Generate result figures from the gateway's routing logs.

Reads logs/route_log.jsonl (one JSON object per request) and renders:
  * results/routing_results.png — traffic split, latency and token distributions
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

rows = []
for line in (ROOT / "logs" / "route_log.jsonl").read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        continue
    if record.get("ok"):
        rows.append(record)

by_route: dict[str, list[dict]] = defaultdict(list)
for record in rows:
    by_route[record.get("route", "unknown")].append(record)

ROUTE_COLORS = {"default": "#2b4f9e", "simple_task": "#e8a33d",
                "cache": "#7aa7e8", "unknown": "#6b7280"}

names = sorted(by_route.keys(), key=lambda n: -len(by_route[n]))
colors = [ROUTE_COLORS.get(n, "#6b7280") for n in names]

fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

# --- 1. traffic split ------------------------------------------------------
counts = [len(by_route[n]) for n in names]
bars = axes[0].bar(names, counts, color=colors)
axes[0].bar_label(bars, fontsize=11, padding=2)
axes[0].set_title(f"Requests by route  (n = {len(rows)})", fontsize=11)
axes[0].set_ylabel("Requests")
axes[0].grid(axis="y", alpha=0.25)
axes[0].set_axisbelow(True)

# --- 2. latency ------------------------------------------------------------
for name, color in zip(names, colors):
    latencies = [r["latency"] for r in by_route[name] if "latency" in r]
    axes[1].hist(latencies, bins=40, alpha=0.65, label=name, color=color)
axes[1].set_xlabel("Latency [s]")
axes[1].set_title("Latency distribution by route", fontsize=11)
axes[1].legend()
axes[1].grid(alpha=0.25)
axes[1].set_axisbelow(True)

# --- 3. input tokens -------------------------------------------------------
for name, color in zip(names, colors):
    tokens = [r.get("in_tokens", 0) for r in by_route[name]]
    axes[2].hist(tokens, bins=40, alpha=0.65, label=name, color=color)
axes[2].set_xlabel("Input tokens")
axes[2].set_title("Input-token distribution by route", fontsize=11)
axes[2].legend()
axes[2].grid(alpha=0.25)
axes[2].set_axisbelow(True)

fig.tight_layout()
out = RESULTS / "routing_results.png"
fig.savefig(out, dpi=150)
plt.close(fig)

print(f"✅ {out}")
print(f"   total requests: {len(rows)}")
for name in names:
    lat = sorted(r["latency"] for r in by_route[name] if "latency" in r)
    tin = sorted(r.get("in_tokens", 0) for r in by_route[name])
    p50_lat = lat[len(lat) // 2] if lat else float("nan")
    p50_tok = tin[len(tin) // 2] if tin else 0
    print(f"   {name:14s} n={len(by_route[name]):4d}  "
          f"median latency={p50_lat:6.2f}s  median in_tokens={p50_tok:6.0f}")
