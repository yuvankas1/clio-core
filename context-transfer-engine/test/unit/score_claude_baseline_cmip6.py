#!/usr/bin/env python3
"""Score the Claude tool-using baseline against cmip6_queries.json.

Reads `fresh_results_baseline_cmip6.json` produced by the protocol in
PROTOCOL_BASELINE_CMIP6.md, compares each `answer_path` against the
expected substring(s) for that query, and prints:

  - hits/50 (single number, comparable to ES/Neo4j/BM25/Qdrant
    backends in fresh_results_k_cmip6.json)
  - per-miss list with expected vs. answered paths
  - aggregated token cost (sum of input+output) if /cost numbers
    were captured

The matching rule is the same as run_k_sweep_all.py:508 — the expected
substring must appear in the answered path. With k=1 (single answer
per query) this collapses to "did Claude pick a path containing the
target identifier?"
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
QUERIES_PATH = Path("/u/rpawar/delta_sweep/cmip6_queries.json")
RESULTS_PATH = HERE / "fresh_results_baseline_cmip6.json"


def main():
    if not RESULTS_PATH.exists():
        print(f"ERROR: {RESULTS_PATH} not found", file=sys.stderr)
        print("Run the protocol in PROTOCOL_BASELINE_CMIP6.md first and "
              "save Claude's JSON output to that path.", file=sys.stderr)
        sys.exit(1)

    qb = json.loads(QUERIES_PATH.read_text())
    expected_by_id = {q["id"]: q["expected"] for q in qb["queries"]}
    tier_by_id     = {q["id"]: q.get("tier", "?") for q in qb["queries"]}
    n = len(qb["queries"])

    res = json.loads(RESULTS_PATH.read_text())
    assert res.get("n_queries") == n, f"expected {n} queries, got {res.get('n_queries')}"

    tier_total = {"A": 0, "B": 0, "C": 0}
    tier_hits  = {"A": 0, "B": 0, "C": 0}
    misses = []
    total_tokens_in = 0
    total_tokens_out = 0
    have_tokens = True

    for r in res["results"]:
        qid = r["id"]
        ans = r.get("answer_path") or ""
        expected_subs = expected_by_id[qid]
        tier = tier_by_id.get(qid, "?")
        is_hit = any(sub in ans for sub in expected_subs)
        if tier in tier_total:
            tier_total[tier] += 1
            if is_hit:
                tier_hits[tier] += 1
        if not is_hit:
            misses.append((qid, tier, expected_subs[0], ans or "<null>"))

        ti0 = r.get("tokens_in_start")
        ti1 = r.get("tokens_in_end")
        to0 = r.get("tokens_out_start")
        to1 = r.get("tokens_out_end")
        if None in (ti0, ti1, to0, to1):
            have_tokens = False
        else:
            total_tokens_in += (ti1 - ti0)
            total_tokens_out += (to1 - to0)

    total_hits = sum(tier_hits.values())
    print(f"Claude {res['agent']} {res['arm']}: {total_hits}/{n} hits "
          f"({100.0*total_hits/n:.1f}%)")
    print(f"  Tier A (filename, n={tier_total['A']:>2}): "
          f"{tier_hits['A']:>2}/{tier_total['A']:>2}")
    print(f"  Tier B (metadata, n={tier_total['B']:>2}): "
          f"{tier_hits['B']:>2}/{tier_total['B']:>2}")
    print(f"  Tier C (content,  n={tier_total['C']:>2}): "
          f"{tier_hits['C']:>2}/{tier_total['C']:>2}")
    if have_tokens:
        per_q_in = total_tokens_in / n
        per_q_out = total_tokens_out / n
        print(f"\nTokens: {total_tokens_in:,} in + {total_tokens_out:,} out "
              f"=  {per_q_in:.0f} in / {per_q_out:.0f} out per query")
    else:
        print("\nTokens: not captured (some /cost fields were null)")

    if misses:
        print(f"\n{len(misses)} misses (grouped by tier):")
        for tier in ['A', 'B', 'C']:
            tmiss = [m for m in misses if m[1] == tier]
            if tmiss:
                print(f"  Tier {tier} ({len(tmiss)} misses):")
                for qid, _, exp, ans in tmiss:
                    print(f"    q{qid:>2}: expected '{exp}'  got  '{ans}'")


if __name__ == "__main__":
    main()
