#!/usr/bin/env python3
"""Render fig_chroma_tokens.png: per-query token cost for Chroma context-1,
colored by hit/miss.

Reads fresh_results_context1.json.
"""

import json
import os
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    with open(os.path.join(HERE, "fresh_results_context1.json")) as f:
        d = json.load(f)
    results = d["results"]

    # Per-query totals
    ids, totals, hits, calls, walls = [], [], [], [], []
    for r in results:
        prompt_sum = 0
        compl_chars = 0
        for t in r["trace_summary"]:
            if isinstance(t, dict) and "prompt_tokens" in t:
                prompt_sum += t.get("prompt_tokens") or 0
                compl_chars += t.get("generated_chars") or 0
        compl_tok = round(compl_chars / 3.7)   # rough est
        ids.append(f"q{r['id']}")
        totals.append(prompt_sum + compl_tok)
        hits.append(bool(r.get("hit")))
        calls.append(r.get("tool_calls", 0))
        walls.append(r.get("elapsed_s", 0.0))

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})
    ax = axes[0]

    bar_colors = ["#2a8a3a" if h else "#a64545" for h in hits]
    bars = ax.bar(ids, totals, color=bar_colors, edgecolor="black", linewidth=0.5)
    for b, v, h, c in zip(bars, totals, hits, calls):
        mark = "OK" if h else "X"
        ax.text(b.get_x() + b.get_width()/2, v + max(totals)*0.01,
                f"{v//1000}.{(v%1000)//100}k\n{mark}",
                ha="center", va="bottom", fontsize=9)

    mean_hit = sum(v for v, h in zip(totals, hits) if h) / max(1, sum(hits))
    mean_miss = sum(v for v, h in zip(totals, hits) if not h) / max(1, sum(1 for h in hits if not h))
    n_hits = sum(hits)
    n_miss = len(hits) - n_hits
    grand_total = sum(totals)

    ax.axhline(y=mean_hit,  color="#2a8a3a", linestyle="--", linewidth=1.0, alpha=0.6,
               label=f"hit mean ({mean_hit:,.0f} tokens, n={n_hits})")
    ax.axhline(y=mean_miss, color="#a64545", linestyle="--", linewidth=1.0, alpha=0.6,
               label=f"miss mean ({mean_miss:,.0f} tokens, n={n_miss})")
    ax.set_ylabel("LLM tokens consumed (prompt + completion, per query)")
    ax.set_ylim(0, max(totals) * 1.18)
    ax.set_title(
        f"Chroma context-1: per-query LLM token cost (best-effort harness)\n"
        f"Total: {grand_total:,} tokens for 18 queries -> {grand_total/18:,.0f} mean. "
        f"Misses average {mean_miss/mean_hit:.1f}x the tokens of hits "
        f"(the agent keeps searching when it can't commit).",
        fontsize=11)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(loc="upper right", fontsize=10)

    # Bottom panel: tool calls per query
    ax2 = axes[1]
    ax2.bar(ids, calls, color=bar_colors, edgecolor="black", linewidth=0.5)
    for x, v in zip(ids, calls):
        ax2.text(x, v + 0.05, str(v), ha="center", va="bottom", fontsize=9)
    ax2.set_ylabel("tool calls")
    ax2.set_ylim(0, max(calls) + 1)
    ax2.set_xlabel("query")
    ax2.grid(axis="y", linestyle=":", alpha=0.4)
    total_calls = sum(calls)
    ax2.text(0.5, 0.85,
             f"total tool calls: {total_calls}  -- "
             f"{total_calls/n_hits:.2f} per hit  "
             f"(Acropolis routed: 1.0 / hit; baseline: 2.0 / hit)",
             transform=ax2.transAxes, ha="center", fontsize=10,
             bbox=dict(boxstyle="round", facecolor="#f5f5f5", edgecolor="#aaa"))

    fig.tight_layout()
    out_path = os.path.join(HERE, "fig_chroma_tokens.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
