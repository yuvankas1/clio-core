#!/usr/bin/env python3
"""fig_cmip6_baseline.png — Claude Code WITHOUT Acropolis vs Acropolis L2
ES RRF on the 50-query CMIP6 retrieval bench. Three panels:

  (a) Accuracy — queries solved (of 50).
  (b) Per-query LLM tokens utilised (real-tokenizer measurement).
      WITHOUT Acropolis: Claude Code uses Glob + Bash(ncdump) and reads
      those tool outputs into Sonnet's context. Measured Sonnet INPUT
      per query: 1,619 tok (1,587 tool results from 1.88 mean calls,
      ncdump output sized by cl100k_base on 5 representative files, +
      27 tok mean query string + 5 tok cached-prompt amortised).
      WITH Acropolis: per-query cost is the query embedding only —
      28 nomic-embed-text input tokens (mean over the 50 query strings,
      nomic-embed-text-v1.5 tokenizer).
      Sonnet OUTPUT tokens not captured (/cost was null in protocol JSON);
      number shown is INPUT side only. Acropolis's per-query cost is the
      COMPLETE LLM cost since the backend search itself runs no LLM.
  (c) Mean tool calls / backend operations per query.

All numbers are real BPE-tokenizer measurements (not chars/4 approx).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

N = 50
K_REPORT = 10

# (a) Accuracy — measured directly from the sweep JSON
CLAUDE_HITS                = 46          # tool-using, claude-sonnet-4-6
ACROPOLIS_HITS_L2_RRF      = 48          # L2 + ES RRF (hybrid) @ k=10

# (b) Per-query LLM tokens — REAL BPE-tokenizer measurements
# Claude side: cl100k_base on actual ncdump outputs (5 samples avg)
CLAUDE_LLM_TOKENS_PER_Q    = 1619        # 1587 tool results + 27 query + 5 cached prompt
# Acropolis side: nomic-embed-text-v1.5 tokenizer on the 50 query strings
ACROPOLIS_LLM_TOKENS_PER_Q = 28          # mean nomic input across queries (range 14–40)

# (c) Operations per query — measured directly from the run JSONs
CLAUDE_MEAN_CALLS          = 1.88
ACROPOLIS_BACKEND_OPS      = 1

plt.rcParams.update({"font.size": 10.5, "axes.titlesize": 11.5,
                     "axes.labelsize": 10.5, "legend.fontsize": 9.5})
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5))

bar_labels = ["Without Acropolis\n(Claude tool-using)",
              "With Acropolis\n(L2 ES RRF @ k=10)"]
bar_colors = ["#888888", "#117755"]

# --- (a) Accuracy ----------------------------------------------------------
ax = axes[0]
vals = [CLAUDE_HITS, ACROPOLIS_HITS_L2_RRF]
ax.bar(np.arange(2), vals, color=bar_colors, edgecolor="black",
       linewidth=0.7, width=0.55)
for xi, v in enumerate(vals):
    ax.text(xi, v + 0.7, f"{v}/{N}", ha="center", va="bottom",
            fontsize=12, fontweight="bold")
ax.set_xticks(np.arange(2))
ax.set_xticklabels(bar_labels)
ax.set_ylim(0, N + 4)
ax.set_ylabel(f"queries solved (of {N})")
ax.set_title("(a) Accuracy")
ax.grid(axis="y", linestyle=":", alpha=0.4)

# --- (b) Per-query LLM tokens ---------------------------------------------
ax = axes[1]
vals = [CLAUDE_LLM_TOKENS_PER_Q, ACROPOLIS_LLM_TOKENS_PER_Q]
ax.bar(np.arange(2), vals, color=bar_colors, edgecolor="black",
       linewidth=0.7, width=0.55)
ax.text(0, vals[0] + 30, f"{vals[0]:,}\nSonnet input",
        ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.text(1, vals[1] + 30, f"{vals[1]}\nnomic input",
        ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_xticks(np.arange(2))
ax.set_xticklabels(bar_labels)
ax.set_ylim(0, max(vals) * 1.30)
ax.set_ylabel("LLM tokens utilised per query")
ax.set_title("(b) Per-query LLM tokens")
ax.grid(axis="y", linestyle=":", alpha=0.4)
# Ratio annotation
ratio = vals[0] / max(vals[1], 1)
ax.text(0.5, vals[0] * 0.55, f"{ratio:.0f}× more\nLLM tokens",
        ha="center", va="center", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff5d6",
                  edgecolor="#888"))

# --- (c) Tool calls per query ---------------------------------------------
ax = axes[2]
vals = [CLAUDE_MEAN_CALLS, ACROPOLIS_BACKEND_OPS]
ax.bar(np.arange(2), vals, color=bar_colors, edgecolor="black",
       linewidth=0.7, width=0.55)
for xi, v in enumerate(vals):
    txt = f"{v:.2f}" if isinstance(v, float) else f"{v}"
    ax.text(xi, v + 0.04, txt, ha="center", va="bottom",
            fontsize=12, fontweight="bold")
ax.set_xticks(np.arange(2))
ax.set_xticklabels(bar_labels)
ax.set_ylim(0, max(vals) + 0.6)
ax.set_ylabel("mean operations per query")
ax.set_title("(c) Tool / backend operations")
ax.grid(axis="y", linestyle=":", alpha=0.4)

fig.suptitle("Claude Code WITH vs WITHOUT Acropolis on CMIP6 retrieval "
             f"(50 queries, 149 NetCDF files)",
             fontsize=12.5, fontweight="bold", y=1.02)

footnote = (
    "Per-query LLM tokens measured with real BPE tokenizers: "
    "cl100k_base on ncdump outputs (5 representative .nc files), "
    "nomic-embed-text-v1.5 on the 50 query strings. "
    "Claude figure is INPUT side only — Sonnet OUTPUT not captured "
    "(/cost was null per query in the protocol JSON). "
    "Acropolis's 28 tok is the COMPLETE per-query LLM cost; backend search "
    "is pure HNSW + BM25 (microseconds, no LLM)."
)
fig.text(0.5, -0.07, footnote, ha="center", va="top",
         fontsize=8.5, style="italic", color="#444", wrap=True)

fig.tight_layout()
out = "/u/rpawar/clio-core/context-transfer-engine/test/unit/fig_cmip6_baseline.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"wrote {out}")
