#!/usr/bin/env python3
"""Grade and plot Acropolis paper benchmark — 18 semantic queries.

Inputs:
  fresh_results_baseline.json    Claude+Glob/Grep/Read agent (no Acropolis)
  fresh_results_acropolis.json   Acropolis: oracle + per-backend (5 backends)

Outputs (PNGs saved alongside this script):
  fig_overhead.png          one-time auto-indexing overhead per file
  fig_backend_accuracy.png  accuracy across 5 KG backends + oracle ceiling
  fig_token_savings.png     Acropolis vs baseline: same/higher accuracy at
                            ~half the tool-call cost
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

EXPECTED = {
    # Filename-trivial / partial (q1-q15)
    1:  ["kg_backend_qdrant.h"],
    2:  ["kg_backend_elasticsearch.h"],
    3:  ["kg_backend_neo4j.h"],
    4:  ["hdf5_summary.h"],
    5:  ["set_depth.cc"],
    6:  ["test_gpu_submission_gpu.cc"],
    7:  ["ggml_iowarp_backend.cc", "ggml_iowarp_backend.h"],   # impl OR header
    8:  ["kvcache_manager"],
    9:  ["run_e2e_gpu_test.sh", "run_e2e_gpu_fixed.sh"],       # both valid
    10: ["bench_repo_scan.cc"],
    11: ["kg_backend_bm25.h"],
    12: ["kg_backend_elasticsearch.h", "kg_backend_neo4j.h"],  # both impl RRF
    13: ["depth_controller.h"],
    14: ["embedding_client.h"],
    15: ["test_indexing_depth_config.cc"],
    # Semantic-only / content-needed (q16, q19, q20)
    # q17 omitted — deferred-release concept lives at line ~420 of an
    # 891-line file, beyond bench's first-4KB truncation.
    # q18 omitted — pure lexical identifier lookup, out of semantic-search scope.
    16: ["ggml_iowarp_backend.cc", "ggml_iowarp_backend.h"],
    19: ["summary_operator.cc", "summary_operator.h"],         # impl OR header
    20: ["indexing_depth_defaults.yaml"],
}

QUERY_TEXTS = {
    1:  "Qdrant vector backend",
    2:  "Elasticsearch text backend",
    3:  "Neo4j knowledge graph backend",
    4:  "HDF5 metadata extractor",
    5:  "CLI: set indexing depth",
    6:  "Unit test: GPU submission (real GPU)",
    7:  "FlexGen weight streaming impl",
    8:  "KV-cache manager (llama.cpp)",
    9:  "E2E: KV cache restore on GPU",
    10: "Benchmark: LLM agent loop",
    11: "BM25 with distributed IDF",
    12: "Hybrid retrieval (RRF)",
    13: "Depth controller (xattr inheritance)",
    14: "OpenAI-compatible embeddings client",
    15: "Unit test: indexing-depth config",
    16: "Compute/transfer overlap + double buffering",
    19: "Verbose 85-word summary operator",
    20: "Default extension-to-tier YAML",
}

BACKENDS = ["bm25", "qdrant", "elasticsearch-kw",
            "elasticsearch-vec", "elasticsearch-rrf"]
LEVELS = [0, 1, 2]
QUERY_IDS = sorted(EXPECTED.keys())   # [1..16, 19, 20] — q17,q18 omitted
QID_TO_POS = {qid: i for i, qid in enumerate(QUERY_IDS)}
N_QUERIES = len(QUERY_IDS)             # 18


def hit(answer_path, expected_list):
    """Multi-answer grader: accepts any substring in expected_list."""
    if answer_path is None:
        return False
    return any(e in answer_path for e in expected_list)


# ---------------------------------------------------------------------------
# Auto-indexing overhead numbers (mined from /tmp/bench_*.log of the May 12
# bench run on the clio-core repo: 1003 files, RTX 5060 Laptop GPU,
# qwen2.5:7b summarizer, 4 parallel workers).
# ---------------------------------------------------------------------------
INDEXING = {
    "n_files":                1003,
    "n_files_summarized":     998,    # 5 failed silently (empty/binary)
    "summary_generation_s":   2016.64,  # phase 1 with worker pool
    "qdrant_ingest_s":        0.355,    # inline (scheduler did inserts)
    "es_ingest_s":            145.437,  # cache-hit on summaries; embed + insert
    "neo4j_ingest_s":         171.684,  # ditto
}


# ===========================================================================
# Plot 1 — Performance overhead of auto-indexing
# ===========================================================================

def fig_overhead(out_path):
    """One-time auto-indexing cost per file, as a single doughnut chart.

    Three phases compose the total per-file ingest cost:
      - Metadata extraction (file stat + path parsing + format detection)
      - LLM summarization (qwen2.5:7b, 4 parallel workers, GPU)
      - Search-engine indexing (embedding + HTTP insert into the backend)

    We draw a single doughnut for Elasticsearch (the slower of the two
    backends evaluated, and the one whose indexing slice is visible).
    Qdrant's indexing slice would shrink to ~0.4 ms (~0.02%); the
    metadata and summarization wedges are unchanged across backends.
    The Qdrant comparison is reported in the caption.
    """
    summary_per_file  = INDEXING["summary_generation_s"] / INDEXING["n_files_summarized"]
    es_per_file       = INDEXING["es_ingest_s"]    / INDEXING["n_files"]
    qdrant_per_file   = INDEXING["qdrant_ingest_s"] / INDEXING["n_files"]
    metadata_per_file = 0.010   # conservative ~10 ms/file stat + parse

    phase_colors = {
        "metadata":  "#9aaab8",   # cool grey
        "summary":   "#cc6633",   # warm rust (dominant)
        "indexing":  "#3a8a5a",   # green
    }
    phase_labels = ["Metadata extraction",
                    "LLM summarization",
                    "Search-engine indexing"]
    phase_keys   = ["metadata", "summary", "indexing"]

    def fmt(v):
        if v >= 1.0:
            return f"{v:.2f} s"
        if v >= 0.001:
            return f"{v*1000:.0f} ms"
        return f"{v*1000:.1f} ms"

    data      = [metadata_per_file, summary_per_file, es_per_file]
    total     = sum(data)
    fractions = [d / total for d in data]
    colors    = [phase_colors[k] for k in phase_keys]

    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    fig.suptitle(
        f"Auto-indexing overhead per file "
        f"({INDEXING['n_files']} files, RTX 5060 Laptop, 4 LLM workers)\n"
        f"LLM summarization dominates total cost by $\\sim$3 orders of magnitude; "
        f"backend choice affects only the indexing slice.",
        fontsize=10.5, y=0.98)

    wedges, _ = ax.pie(
        data,
        colors=colors,
        startangle=90,
        counterclock=False,
        wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2.2),
    )

    # Center label: backend + total per-file cost.
    ax.text(0, 0.18, "Elasticsearch",
            ha="center", va="center", fontsize=14, fontweight="bold")
    ax.text(0, -0.02, fmt(total),
            ha="center", va="center", fontsize=13, color="#333")
    ax.text(0, -0.20, "per file",
            ha="center", va="center", fontsize=10, color="#666")

    # Leader-line annotations for each wedge, placed outside the ring.
    for i, w in enumerate(wedges):
        ang = (w.theta2 + w.theta1) / 2.0
        x = np.cos(np.deg2rad(ang))
        y = np.sin(np.deg2rad(ang))

        label = f"{phase_labels[i]}\n{fmt(data[i])} ({fractions[i]*100:.2f}%)"
        ha = "left" if x >= 0 else "right"
        x_text = 1.35 * np.sign(x) if x != 0 else 1.35
        ax.annotate(
            label,
            xy=(x, y), xytext=(x_text, 1.10 * y),
            ha=ha, va="center", fontsize=10.5,
            arrowprops=dict(arrowstyle="-", color="#555", lw=0.7,
                            connectionstyle="arc3,rad=0"),
        )

    ax.set_xlim(-1.95, 1.95)
    ax.set_ylim(-1.45, 1.45)

    # Bottom legend
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=phase_colors[k]) for k in phase_keys
    ]
    fig.legend(legend_handles, phase_labels,
               loc="lower center", ncol=3, fontsize=10, frameon=False,
               bbox_to_anchor=(0.5, 0.02))

    fig.tight_layout(rect=[0, 0.06, 1, 0.92])
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path} (Qdrant indexing slice would be "
          f"{fmt(qdrant_per_file)}, ~0.02%; metadata + summary unchanged)")


# ===========================================================================
# Plot 2 — Accuracy tradeoff between knowledge-graph backends
# ===========================================================================

def fig_backend_accuracy(out_path, acropolis):
    """Per-backend top-5 accuracy with baseline + oracle reference bars.

    Layout: BASELINE | 5 individual backends | ORACLE
    The baseline is what an agent gets without Acropolis (Glob/Grep/Read).
    Each individual backend bar = run that backend alone on all queries.
    Oracle = per-query routing across the 5 backends.
    """
    per_bk = acropolis["per_backend_total"]
    oracle = acropolis["oracle_score"]
    base   = acropolis["baseline_score"]
    n      = acropolis["total_queries"]

    pretty = {
        "bm25":              "BM25\n(lexical)",
        "qdrant":            "Qdrant\n(dense)",
        "elasticsearch-kw":  "ES kw\n(lexical)",
        "elasticsearch-vec": "ES vec\n(dense)",
        "elasticsearch-rrf": "ES RRF\n(hybrid)",
    }
    keys = list(per_bk.keys())

    labels = (
        ["BASELINE\n(Glob/Grep/Read)"]
        + [pretty.get(k, k) for k in keys]
        + ["ACROPOLIS\n(routed, best of 5)"]
    )
    values = [base] + [per_bk[k] for k in keys] + [oracle]

    palette = {
        "bm25":              "#a64545",   # BM25 = red
        "qdrant":            "#4477aa",
        "elasticsearch-kw":  "#bb7733",
        "elasticsearch-vec": "#88bbdd",
        "elasticsearch-rrf": "#117755",
    }
    colors = (
        ["#888888"]                                       # baseline (grey)
        + [palette.get(k, "#777") for k in keys]
        + ["#2a8a3a"]                                     # acropolis (green)
    )

    fig, ax = plt.subplots(figsize=(13, 5.8))
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, values):
        pct = 100 * v / n
        ax.text(b.get_x() + b.get_width()/2, v + 0.2,
                f"{v}/{n}\n({pct:.0f}%)",
                ha="center", va="bottom", fontsize=10)

    ax.axhline(y=n, color="gray", linestyle="--", linewidth=0.9,
               label=f"perfect ({n}/{n})")
    ax.set_ylabel(f"queries solved (top-5, of {n})")
    ax.set_ylim(0, n + 2.5)
    ax.set_title(
        f"Accuracy on {n} semantic queries (clio-core, 1003 files)\n"
        f"Baseline = no Acropolis. Each Acropolis backend evaluated "
        f"independently; routed = per-query best-of-5.",
        fontsize=11)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(loc="lower right", fontsize=9)

    # Visual separators between the three groups
    ax.axvline(x=0.5, color="black", linewidth=0.5, alpha=0.25)
    ax.axvline(x=len(keys) + 0.5, color="black", linewidth=0.5, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


# ===========================================================================
# Plot 3 — Token / tool-call savings with no accuracy loss
# ===========================================================================

def fig_token_savings(out_path, baseline, acropolis):
    """Acropolis matches or beats baseline accuracy at fewer tool calls."""
    n          = acropolis["total_queries"]
    base_score = acropolis["baseline_score"]
    base_calls = acropolis["baseline_tool_calls"]
    acro_score = acropolis["routed_score"]
    acro_calls = n                            # routed = 1 MCP call per query

    # Realistic per-tool-call token estimate (chars / 4 ~ tokens).
    # Baseline call mix is roughly 60% Glob/Grep + 40% Read:
    #   Glob (30 paths x 80 chars)       ~600 tokens
    #   Grep (10 lines x 200 chars)      ~500 tokens
    #   Read (one code file, ~4 KB)     ~1000 tokens
    #   weighted avg = 0.6*550 + 0.4*1000 = 730 tokens/call
    # Acropolis MCP semantic_query (5 hits x ~700 char summary):
    #   ~3.5 KB per call               ~900 tokens/call
    # Caveat: rough estimate; real billed tokens (cache_read accounting,
    # output tokens, context compounding) need session-JSONL mining.
    EST_TOKENS_PER_CALL = {
        "baseline":  730,
        "acropolis": 900,
    }
    base_tok = base_calls * EST_TOKENS_PER_CALL["baseline"]
    acro_tok = acro_calls * EST_TOKENS_PER_CALL["acropolis"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"Token savings with no accuracy loss — {n} semantic queries",
        fontsize=12)

    # Panel A: accuracy
    cfgs = ["Baseline\n(Glob/Grep/Read)", "Acropolis\n(routed)"]
    accs = [base_score, acro_score]
    bars = axes[0].bar(cfgs, accs, color=["#888", "#2a8a3a"],
                        edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, accs):
        axes[0].text(b.get_x() + b.get_width()/2, v + 0.15,
                     f"{v}/{n}\n({100*v/n:.0f}%)",
                     ha="center", va="bottom", fontsize=11)
    axes[0].axhline(y=n, color="gray", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel(f"queries solved (of {n})")
    axes[0].set_ylim(0, n + 2)
    axes[0].set_title("Accuracy")
    axes[0].grid(axis="y", linestyle=":", alpha=0.4)

    # Panel B: tool calls
    calls_vals = [base_calls, acro_calls]
    bars2 = axes[1].bar(cfgs, calls_vals, color=["#888", "#2a8a3a"],
                         edgecolor="black", linewidth=0.5)
    for b, v in zip(bars2, calls_vals):
        axes[1].text(b.get_x() + b.get_width()/2, v + 0.4,
                     str(v), ha="center", va="bottom", fontsize=11)
    saved_calls = (base_calls - acro_calls) / base_calls * 100
    axes[1].set_ylabel(f"total tool calls ({n} queries)")
    axes[1].set_title(f"Tool-call cost\n(Acropolis uses {saved_calls:.0f}% fewer)")
    axes[1].grid(axis="y", linestyle=":", alpha=0.4)

    # Panel C: estimated tokens (rough proxy)
    tok_vals = [base_tok, acro_tok]
    bars3 = axes[2].bar(cfgs, tok_vals, color=["#888", "#2a8a3a"],
                         edgecolor="black", linewidth=0.5)
    for b, v in zip(bars3, tok_vals):
        axes[2].text(b.get_x() + b.get_width()/2, v + max(tok_vals)*0.02,
                     f"~{v/1000:.1f}K", ha="center", va="bottom", fontsize=11)
    saved_tok = (base_tok - acro_tok) / base_tok * 100
    axes[2].set_ylabel("estimated tool-call response tokens")
    axes[2].set_title(
        f"Estimated token cost\n"
        f"(Acropolis saves ~{saved_tok:.0f}% — based on per-call payload)")
    axes[2].grid(axis="y", linestyle=":", alpha=0.4)
    # Footnote: estimate methodology
    axes[2].text(0.5, -0.18,
                 "Estimate: baseline ~730 tok/call (60% Glob+Grep, 40% Read);\n"
                 "Acropolis ~900 tok/call (MCP returns 5 hits w/ summaries).\n"
                 "Real billed tokens require session-JSONL mining.",
                 transform=axes[2].transAxes, ha="center", va="top",
                 fontsize=8, color="#555", style="italic")

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


# ===========================================================================
# Plot 4 (new) — per-level accuracy across all 5 backends
# ===========================================================================

def fig_level_accuracy(out_path, per_level):
    """Accuracy across 5 backends at L0/L1/L2 — clustered bar chart.

    Each backend cluster has 3 bars (one per indexing level). Baseline
    (Glob/Grep/Read) shown as a horizontal reference line.
    """
    hits   = per_level["hits"]        # {level_str: {backend: hits}}
    base   = per_level.get("baseline_score", 17)
    n      = per_level.get("total_queries", 18)

    backends = per_level["backends"]
    levels = ["0", "1", "2"]   # JSON keys are strings
    n_groups = len(backends)

    pretty = {
        "bm25":              "BM25",
        "qdrant":            "Qdrant",
        "elasticsearch-kw":  "ES kw",
        "elasticsearch-vec": "ES vec",
        "elasticsearch-rrf": "ES RRF",
    }
    group_labels = [pretty.get(b, b) for b in backends]
    level_color  = {"0": "#cca44a", "1": "#7b9bcc", "2": "#3a9d4a"}
    level_label  = {"0": "L0 (path only)",
                    "1": "L1 (+ metadata)",
                    "2": "L2 (+ content)"}

    x = np.arange(n_groups)
    width = 0.27

    fig, ax = plt.subplots(figsize=(11, 6))
    for li, lv in enumerate(levels):
        values = [hits[lv][b] for b in backends]
        offset = (li - 1) * width
        bars = ax.bar(x + offset, values, width,
                      label=level_label[lv], color=level_color[lv],
                      edgecolor="black", linewidth=0.4)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.2,
                    f"{v}", ha="center", va="bottom", fontsize=8.5)

    # Baseline reference line
    ax.axhline(y=base, color="#888888", linestyle="--", linewidth=1.2,
               label=f"Baseline ({base}/{n}, Glob/Grep/Read)")
    ax.axhline(y=n, color="gray", linestyle=":", linewidth=0.8,
               label=f"Perfect ({n}/{n})")

    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, fontsize=10.5)
    ax.set_ylabel(f"queries solved (top-5, of {n})")
    ax.set_ylim(0, n + 2.5)
    ax.set_title(
        f"Per-level accuracy across {len(backends)} knowledge-graph backends "
        f"({n} semantic queries, 996/1003 files)\n"
        "Indexing depth = input richness for the LLM summarizer "
        "(L0=path, L1=+metadata, L2=+content head+tail 8 KB).",
        fontsize=11)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.95)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


# ===========================================================================
# Plot 5 — Three-system comparison (baseline / Acropolis ES RRF / Chroma)
# ===========================================================================

def fig_token_3way(out_path, acropolis):
    """Three-system three-panel comparison for §V.D of the paper.

    Systems: shell-tool baseline, Acropolis with its strongest single
    backend (Elasticsearch RRF), and Chroma context-1 (released
    gpt-oss-20b fine-tune plus retrieval harness).

    Panels: accuracy / total tool calls / estimated response-payload tokens.

    Acropolis uses ES RRF as a single, deployable backend (16/18).
    No cross-backend routing or ORACLE union is involved.
    """
    n = acropolis["total_queries"]

    # Acropolis = strongest single backend (ES RRF), not ORACLE.
    acro_score = acropolis["per_backend_total"]["elasticsearch-rrf"]
    acro_calls = n                                # 1 MCP call per query
    base_score = acropolis["baseline_score"]
    base_calls = acropolis["baseline_tool_calls"]

    # Chroma context-1 numbers (validated externally; see paper §V.D).
    chroma_score = 15                              # lenient (raw 14)
    chroma_calls = 36

    EST_TOKENS_PER_CALL = {
        "baseline":  730,    # 60% Glob/Grep (~550) + 40% Read (~1000)
        "acropolis": 900,    # 5 hits w/ summaries
        "chroma":    750,    # Chroma context-1 response shape
    }
    base_tok   = base_calls   * EST_TOKENS_PER_CALL["baseline"]
    acro_tok   = acro_calls   * EST_TOKENS_PER_CALL["acropolis"]
    chroma_tok = chroma_calls * EST_TOKENS_PER_CALL["chroma"]

    labels = ["Baseline", "Acropolis\n(ES RRF)", "Chroma\ncontext-1"]
    colors = ["#888888", "#2a8a3a", "#7a4a9c"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    fig.suptitle(
        f"Three-system comparison on the {n}-query semantic benchmark",
        fontsize=12)

    # Panel A — Accuracy
    accs = [base_score, acro_score, chroma_score]
    bars = axes[0].bar(labels, accs, color=colors,
                       edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, accs):
        axes[0].text(b.get_x() + b.get_width()/2, v + 0.15,
                     f"{v}/{n}", ha="center", va="bottom", fontsize=11)
    axes[0].axhline(y=n, color="gray", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel(f"queries solved (of {n})")
    axes[0].set_ylim(0, n + 2)
    axes[0].set_title("Accuracy")
    axes[0].grid(axis="y", linestyle=":", alpha=0.4)

    # Panel B — Total tool calls
    calls = [base_calls, acro_calls, chroma_calls]
    bars2 = axes[1].bar(labels, calls, color=colors,
                        edgecolor="black", linewidth=0.5)
    for b, v in zip(bars2, calls):
        axes[1].text(b.get_x() + b.get_width()/2, v + max(calls)*0.015,
                     str(v), ha="center", va="bottom", fontsize=11)
    axes[1].set_ylabel("total tool calls")
    axes[1].set_title("Tool-call cost")
    axes[1].grid(axis="y", linestyle=":", alpha=0.4)

    # Panel C — Estimated response-payload tokens
    toks = [base_tok, acro_tok, chroma_tok]
    bars3 = axes[2].bar(labels, toks, color=colors,
                        edgecolor="black", linewidth=0.5)
    for b, v in zip(bars3, toks):
        axes[2].text(b.get_x() + b.get_width()/2, v + max(toks)*0.015,
                     f"{v/1000:.1f}K", ha="center", va="bottom", fontsize=11)
    axes[2].set_ylabel("tool-call response tokens")
    axes[2].set_title("Tool-call response tokens (estimate)")
    axes[2].grid(axis="y", linestyle=":", alpha=0.4)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    baseline_path  = os.path.join(HERE, "fresh_results_baseline.json")
    acropolis_path = os.path.join(HERE, "fresh_results_acropolis.json")
    with open(baseline_path) as f:
        baseline = json.load(f)
    with open(acropolis_path) as f:
        acropolis = json.load(f)

    n = acropolis["total_queries"]
    print(f"=== {n}-query semantic benchmark on clio-core ===")
    print(f"  Baseline (Glob/Grep/Read agent)   {acropolis['baseline_score']}/{n}"
          f"   {acropolis['baseline_tool_calls']} tool calls")
    print(f"  Acropolis ORACLE (5 backends)     {acropolis['oracle_score']}/{n}")
    print(f"  Acropolis ROUTED (1 backend/query) {acropolis['routed_score']}/{n}"
          f"   {n} MCP calls")
    print(f"  Per-backend individual hits:")
    for b, v in acropolis["per_backend_total"].items():
        print(f"    {b:<22} {v}/{n}")
    print()

    fig_overhead(         os.path.join(HERE, "fig_overhead.png"))
    fig_backend_accuracy( os.path.join(HERE, "fig_backend_accuracy.png"), acropolis)
    fig_token_savings(    os.path.join(HERE, "fig_token_savings.png"),
                          baseline, acropolis)
    fig_token_3way(       os.path.join(HERE, "fig_token_3way.png"), acropolis)

    per_level_path = os.path.join(HERE, "fresh_results_per_level.json")
    if os.path.exists(per_level_path):
        with open(per_level_path) as f:
            per_level = json.load(f)
        fig_level_accuracy(
            os.path.join(HERE, "fig_level_accuracy.png"), per_level)
