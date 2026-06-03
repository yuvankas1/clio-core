#!/usr/bin/env python3
"""Render fig_token_3way.png: 3-system token-cost comparison.

Adds Chroma context-1 to the existing Acropolis baseline + routed comparison.
Two reads of "token cost" because we have different evidence per system:

  Panel A -- Accuracy: queries solved / 18.
  Panel B -- Tool-call cost: total tool calls across 18 queries.
  Panel C -- Estimated tool-call response tokens (apples-to-apples; same
             methodology as plot_fresh_results.fig_token_savings).
  Panel D -- Measured LLM context tokens for Chroma context-1 only; Acropolis
             numbers would require mining session JSONLs from the laptop
             (mine_session_tokens.py). Shown as the *full* cost a researcher
             pays at query time.
"""

import json
import os
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))


def load():
    a = json.load(open(os.path.join(HERE, "fresh_results_acropolis.json")))
    c = json.load(open(os.path.join(HERE, "fresh_results_context1.json")))
    return a, c


def main():
    acro, ctx1 = load()
    n = acro["total_queries"]

    base_score = acro["baseline_score"]
    base_calls = acro["baseline_tool_calls"]
    acro_score = acro["routed_score"]
    acro_calls = n                              # routed = 1 MCP call / query
    ctx1_score = ctx1.get("score_with_q10_credit", ctx1["score"])
    ctx1_raw   = ctx1.get("score_raw", ctx1_score)
    ctx1_calls = sum(r.get("tool_calls", 0) for r in ctx1["results"])

    # ---- estimated response tokens (same methodology as fig_token_savings) ----
    EST_TOK_PER_CALL = {
        "baseline":   730,   # 60% Glob/Grep ~ 550, 40% Read ~ 1000
        "acropolis":  900,   # 5 hits with summaries
        # Chroma's search_corpus returns 5 chunks * up to 800-char snippet ~
        # ~700 tokens; grep returns less; read returns ~1000. Mostly
        # search_corpus in this run.
        "context1":   750,
    }
    base_tok = base_calls * EST_TOK_PER_CALL["baseline"]
    acro_tok = acro_calls * EST_TOK_PER_CALL["acropolis"]
    ctx1_tok_est = ctx1_calls * EST_TOK_PER_CALL["context1"]

    # ---- measured full LLM tokens for Chroma context-1 (from trace) ----
    ctx1_prompt = 0
    ctx1_compl_chars = 0
    for r in ctx1["results"]:
        for t in r["trace_summary"]:
            if isinstance(t, dict):
                ctx1_prompt += t.get("prompt_tokens") or 0
                ctx1_compl_chars += t.get("generated_chars") or 0
    ctx1_compl = round(ctx1_compl_chars / 3.7)
    ctx1_total_llm = ctx1_prompt + ctx1_compl

    # Two-line labels — "Acropolis" is too wide for narrow panels otherwise.
    labels = ["Baseline", "Acropolis\nrouted", "Chroma\ncontext-1"]
    colors = ["#888888", "#2a8a3a", "#7a3f8a"]

    # 2-column-wide research-paper figure: 1×3 grid at 7" × 3.0". The
    # "Full LLM tokens consumed (measured)" panel was dropped — only
    # Chroma's value was mined and reporting it alongside two "not mined"
    # placeholders made the comparison misleading.
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10,
                          "axes.labelsize": 9, "xtick.labelsize": 8,
                          "ytick.labelsize": 8.5})
    fig, axes = plt.subplots(1, 3, figsize=(8.5, 3.2))
    fig.suptitle(
        "Three-system comparison on the 18-query semantic benchmark",
        fontsize=10, y=1.00)

    accs = [base_score, acro_score, ctx1_score]
    calls = [base_calls, acro_calls, ctx1_calls]
    toks = [base_tok, acro_tok, ctx1_tok_est]

    def styled_bars(ax, values, value_fmt, ylabel, title, headroom=1.15):
        bars = ax.bar(labels, values, color=colors, edgecolor="black",
                      linewidth=0.4)
        top = max(values) * headroom if max(values) > 0 else 1
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2,
                    v + max(values) * 0.015,
                    value_fmt(v), ha="center", va="bottom", fontsize=8.5)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(0, top)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        return bars

    # Panel A: accuracy
    styled_bars(axes[0], accs, lambda v: f"{v}/{n}",
                f"queries solved (/{n})", "Accuracy")
    axes[0].set_ylim(0, n + 2)
    axes[0].axhline(y=n, color="gray", linestyle="--", linewidth=0.7)

    # Panel B: tool calls
    styled_bars(axes[1], calls, lambda v: f"{v}",
                "total tool calls", "Tool-call cost")

    # Panel C: estimated tool-call response tokens (apples-to-apples)
    styled_bars(axes[2], toks, lambda v: f"{v/1000:.1f}K",
                "tool-call response tokens",
                "Tool-call response tokens (estimate)")

    fig.tight_layout(rect=[0, 0.02, 1, 0.95], w_pad=1.8)
    out = os.path.join(HERE, "fig_token_3way.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    print()
    print("=== summary numbers ===")
    print(f"           {'acc':>10} {'calls':>10} {'est_resp_tok':>15} {'full_llm_tok':>15}")
    print(f"baseline   {base_score:>4}/{n:<3}  {base_calls:>10} {base_tok:>15,} {'-- (mine JSONLs)':>15}")
    print(f"acropolis  {acro_score:>4}/{n:<3}  {acro_calls:>10} {acro_tok:>15,} {'-- (mine JSONLs)':>15}")
    print(f"context-1  {ctx1_score:>4}/{n:<3}  {ctx1_calls:>10} {ctx1_tok_est:>15,} {ctx1_total_llm:>15,}")


if __name__ == "__main__":
    main()
