#!/usr/bin/env python3
"""Render fig_model_accuracy.png — per-backend accuracy across 4
summarizer LLMs (qwen2.5:7b, qwen2.5-coder:14b, deepseek-coder-v2:16b,
gemma2:27b) run through the production CAE pipeline with the v3 prompt
and nomic-embed-text. gpt-oss:20b excluded — it produced rc=-4 errors on
71% of summarize calls under this pipeline and isn't a fair comparison.
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# Hard-code the results so the figure can render without the per-arm
# JSONs travelling alongside the script.
DATA = [
    ("qwen2.5:7b",            16, 11, 16, 17),
    ("qwen2.5-coder:14b",     17, 14, 17, 17),
    ("deepseek-coder-v2:16b", 16, 11, 17, 18),
    ("gemma2:27b",            16, 10, 16, 16),
]
N = 18  # total queries
LAPTOP_REF = {"es-kw": 17, "es-vec": 14, "es-rrf": 16}
# Claude + Glob/Grep/Read (no Acropolis) — from fresh_results_acropolis.json
CLAUDE_BASELINE = 17


def main():
    # Leftmost bar is the Claude baseline (Glob/Grep/Read, no retrieval).
    # The other 4 are the L2 CAE pipeline + ES RRF, varying summarizer LLM.
    labels = ["Claude\nbaseline"] + [r[0] for r in DATA]
    values = [CLAUDE_BASELINE] + [r[3] for r in DATA]
    # Distinct colors so the baseline reads as a different system.
    bar_colors = ["#888888"] + ["#117755"] * len(DATA)

    x = np.arange(len(labels))

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11,
                          "axes.labelsize": 10, "xtick.labelsize": 9.5,
                          "ytick.labelsize": 9, "legend.fontsize": 9})
    fig, ax = plt.subplots(figsize=(8.0, 3.6))

    bars = ax.bar(x, values, 0.55, color=bar_colors, edgecolor="black",
                  linewidth=0.5)

    # Value labels above each bar
    for b in bars:
        v = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, v + 0.18, f"{v}/{N}",
                ha="center", va="bottom", fontsize=10, weight="bold")

    # Legend handles (matching bar colors)
    import matplotlib.patches as mpatches
    handles = [
        mpatches.Patch(facecolor="#888888", edgecolor="black",
                        linewidth=0.5,
                        label="Claude baseline (Glob/Grep/Read)"),
        mpatches.Patch(facecolor="#117755", edgecolor="black",
                        linewidth=0.5,
                        label="Acropolis ES RRF (hybrid kw+vec) per summarizer LLM"),
    ]
    ax.legend(handles=handles, loc="lower right", framealpha=0.92)


    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(f"queries solved (of {N})")
    ax.set_ylim(0, N + 2)
    ax.set_title(
        f"Claude baseline vs Acropolis ES RRF across summarizer LLMs "
        f"(top-5, {N} queries)",
        pad=8)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    fig.tight_layout()
    out = os.path.join(HERE, "fig_model_accuracy.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    print()
    print(f"  bars rendered: {len(labels)} (1 Claude baseline + {len(DATA)} CAE arms)")
    print(f"  (gpt-oss:20b excluded — 71% summarize failure under this pipeline)")


if __name__ == "__main__":
    main()
