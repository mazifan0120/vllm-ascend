"""Render report figures from the archived, full precision request verdicts."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
rows = json.loads((ROOT / "performance-data.json").read_text())
plt.rcParams.update({"font.size": 10, "svg.fonttype": "none"})
fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
colors = {"feature": "#167d9a", "dcp": "#b5793e"}


def bars(ax, selected, field, title, unit):
    x = np.arange(len(selected))
    for side, key in enumerate(("feature", "dcp")):
        values = [r[key]["ttft_s"]["mean"] if field == "ttft" else r[key][field] for r in selected]
        rects = ax.bar(
            x + (side - 0.5) * 0.36,
            values,
            width=0.36,
            color=colors[key],
            label="KV offload (1M)" if key == "feature" else "Customer DCP2 (512K)",
        )
        ax.bar_label(rects, fmt="%.2f", fontsize=8, padding=3)
    ax.set_xticks(x, [f"C{r['concurrency']} / {r['cache_target_pct']}%" for r in selected])
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_ylabel(unit)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.13)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.2)


bars(axes[0, 0], rows[:2], "ttft", "Mean TTFT: cold external cache (lower is better)", "Seconds")
bars(axes[0, 1], rows[2:], "ttft", "Mean TTFT: ~90% external hits (lower is better)", "Seconds")
bars(axes[1, 0], rows, "tpot_ms", "Mean TPOT (lower is better)", "Milliseconds / token")
bars(axes[1, 1], rows, "output_toks_s", "Output throughput (higher is better)", "Tokens / second")
axes[0, 0].legend(frameon=False, fontsize=9)
fig.suptitle("GLM-5.2 / A3: 55,000 input + 700 output tokens, 112 requests per case", fontsize=14)
fig.get_layout_engine().set(rect=(0, 0.055, 1, 0.945))
fig.text(
    0.5,
    0.018,
    "D TP2/DP16/MTP5 aligned; feature 1M vs customer 512K; software, P settings and KV paths differ.",
    ha="center",
    fontsize=9,
    color="#444444",
)
fig.savefig(ROOT / "performance-overview.png", dpi=180)
fig.savefig(ROOT / "performance-overview.svg")
svg = ROOT / "performance-overview.svg"
svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
plt.close(fig)
