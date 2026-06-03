#!/usr/bin/env python3
"""Render fig_chroma_vs_acropolis.png: 18-query accuracy comparison adding
Chroma context-1 (best-effort harness on chromadb/context-1) alongside the
existing Acropolis backends + baseline + routed.

Reads:
  fresh_results_acropolis.json     -- existing per-backend scores + baseline + oracle
  fresh_results_context1.json      -- this run (raw + q10-credit)
"""

import json
import os
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    with open(os.path.join(HERE, "fresh_results_acropolis.json")) as f:
        acro = json.load(f)
    with open(os.path.join(HERE, "fresh_results_context1.json")) as f:
        ctx1 = json.load(f)

    n = acro["total_queries"]                 # 18
    baseline = acro["baseline_score"]         # 17
    oracle = acro["oracle_score"]             # 18
    per_bk = acro["per_backend_total"]
    ctx1_raw = ctx1.get("score_raw", ctx1["score"])
    ctx1_credit = ctx1.get("score_with_q10_credit", ctx1_raw)

    # Order: baseline | context-1 | five acropolis backends | acropolis routed
    bk_order = ["bm25", "qdrant", "elasticsearch-kw", "elasticsearch-vec", "elasticsearch-rrf"]
    bk_labels = {
        "bm25":              "BM25\n(lexical)",
        "qdrant":            "Qdrant\n(dense)",
        "elasticsearch-kw":  "ES kw\n(lexical)",
        "elasticsearch-vec": "ES vec\n(dense)",
        "elasticsearch-rrf": "ES RRF\n(hybrid)",
    }

    labels = (
        ["BASELINE\n(Claude +\nGlob/Grep/Read)"]
        + ["CHROMA\ncontext-1\n(best-effort)"]
        + [bk_labels[k] for k in bk_order]
    )
    values = (
        [baseline]
        + [ctx1_credit]
        + [per_bk[k] for k in bk_order]
    )
    # Sub-labels: for context-1, annotate raw vs credit.
    sub_annotations = [None] * len(values)
    if ctx1_credit != ctx1_raw:
        sub_annotations[1] = f"(raw {ctx1_raw}/{n})"

    palette = {
        "bm25":              "#a64545",
        "qdrant":            "#4477aa",
        "elasticsearch-kw":  "#bb7733",
        "elasticsearch-vec": "#88bbdd",
        "elasticsearch-rrf": "#117755",
    }
    colors = (
        ["#888888"]                            # baseline grey
        + ["#7a3f8a"]                          # chroma context-1 purple
        + [palette[k] for k in bk_order]
    )

    fig, ax = plt.subplots(figsize=(14, 6.2))
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.5)
    for b, v, sub in zip(bars, values, sub_annotations):
        pct = 100 * v / n
        primary = f"{v}/{n}\n({pct:.0f}%)"
        ax.text(b.get_x() + b.get_width()/2, v + 0.25, primary,
                ha="center", va="bottom", fontsize=10)
        if sub:
            ax.text(b.get_x() + b.get_width()/2, v - 1.4, sub,
                    ha="center", va="top", fontsize=8.5,
                    color="white", weight="bold")

    ax.axhline(y=n, color="gray", linestyle="--", linewidth=0.9,
               label=f"perfect ({n}/{n})")
    ax.set_ylabel(f"queries solved (of {n})")
    ax.set_ylim(0, n + 3)
    ax.set_title(
        f"18-query semantic file retrieval on clio-core (1005 files)\n"
        f"Chroma context-1 = released gpt-oss-20b fine-tune + best-effort harness; "
        f"Acropolis backends index L2 LLM summaries; baseline = no Acropolis.",
        fontsize=11)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(loc="lower right", fontsize=9)

    # Visual separators
    ax.axvline(x=0.5, color="black", linewidth=0.5, alpha=0.25)   # baseline | ctx1
    ax.axvline(x=1.5, color="black", linewidth=0.5, alpha=0.25)   # ctx1 | acropolis backends

    fig.tight_layout()
    out_path = os.path.join(HERE, "fig_chroma_vs_acropolis.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
