# Acropolis protocol — per-backend semantic-search benchmark (L0/L1/L2)

**Scope:** Acropolis is a **semantic file-retrieval** system. Given a
natural-language description of a file's purpose, function, or role, the
system surfaces the right file. **Pure identifier lookups** (the
`grep -r SymbolName` workflow) are **out of scope** — they are a complementary
capability that production systems should layer on top.

**Why this benchmark:** measure per-backend accuracy at all 3 indexing
depths, on a 19-query semantic test set. Tokens-per-query and tool-call
count vs. a Glob/Grep/Read baseline are the headline metrics. Each query
is run at L0, L1, AND L2 inside the same session.

Run **7 separate times** — once per backend — in fresh chats. Edit one
line in the prompt (`backend="<BACKEND>"`) per run.

---

## The 7 backends to run

Run the prompt below 7 times, substituting `<BACKEND>` with each of these
values in order:
  1. `bm25`
  2. `elasticsearch-kw`
  3. `elasticsearch-vec`
  4. `elasticsearch-rrf`
  5. `qdrant`
  6. `neo4j-kw`
  7. `neo4j-rrf`

After each run, save the printed JSON as
`fresh_results_acropolis_<BACKEND>.json`
(e.g. `fresh_results_acropolis_bm25.json`).

---

## Setup (each run)

1. Open a NEW Claude Code chat in workspace `C:\Users\rajni\Documents\GPU_OS`.
2. Acropolis MCP enabled (`/mcp` → connected). Docker Desktop running for
   the ES/Qdrant/Neo4j containers.
3. Paste the prompt below — make sure to substitute `<BACKEND>` first.
4. When done, save the JSON output to the corresponding filename.

---

## Prompt to paste (substitute `<BACKEND>` first)

```
You are running a research benchmark on the Acropolis semantic-search
system. You may use ONLY these tools:
  - mcp__acropolis__semantic_query
  - Read

Do NOT use Glob, Grep, Bash, or any other tool.

For EACH query below, follow this exact procedure for ALL THREE LEVELS:
  1. Call mcp__acropolis__semantic_query three times with backend="<BACKEND>",
     k=5, and level=0, level=1, level=2 (three separate calls per query).
  2. For each level, look at the top-3 returned hits and pick the one
     whose path/summary best fits the query intent. If none clearly fit,
     Read the top-1 candidate to verify before committing.
  3. Total Read budget: up to 3 Reads per query, shared across all 3 levels.
     Reuse Read knowledge across levels when the same candidate appears.
  4. If after Reads no candidate matches at a level, set that level's
     answer_path=null.

For each query, record:
  - id (the integer shown next to each query — note q18 is intentionally absent)
  - n_reads (0..3 — total Reads for this query, shared across levels)
  - configs: list of 3 entries {level, answer_path}

QUERIES (19 semantic queries; q18 omitted as out-of-scope lexical lookup):
  q1:  "Find the Qdrant vector backend implementation file and give its path"
  q2:  "Locate the Elasticsearch full-text search backend"
  q3:  "Show me the Neo4j knowledge graph backend"
  q4:  "Where is the HDF5 metadata extractor?"
  q5:  "Find the CLI tool that sets indexing depth on files"
  q6:  "Locate the unit test that exercises GPU submission on an actual GPU"
  q7:  "Find the implementation file that performs per-layer FlexGen weight streaming through GpuVMM"
  q8:  "Where is the KV-cache manager that interacts with llama.cpp?"
  q9:  "Find the end-to-end script that tests KV cache restore on GPU"
  q10: "Locate the benchmark that scans a repo with an LLM agent loop"
  q11: "Find the file implementing BM25 scoring with distributed IDF synchronization"
  q12: "Where is the hybrid retrieval that uses reciprocal rank fusion?"
  q13: "Locate the depth controller that resolves xattr inheritance across directories"
  q14: "Find the OpenAI-compatible HTTP embeddings client shared across backends"
  q15: "Where is the unit test validating indexing-depth configuration parsing?"
  q16: "Find the implementation that overlaps GPU compute with weight transfer using double buffering during transformer layer execution"
  q17: "Where is the deferred-release fix that prevents the GpuVmm page-overlap bug between adjacent transformer layers?"
  q19: "Find the operator that produces the verbose ~85-word natural-language summary for each file at the deepest indexing tier"
  q20: "Where is the YAML defining the default mapping from file extensions to Acropolis indexing tiers?"

After all 19 queries, print ONE final code block in this exact JSON shape,
with NO commentary:

{
  "agent": "claude-sonnet-4.7",
  "arm": "acropolis_per_backend",
  "config": {"backend": "<BACKEND>", "k": 5},
  "results": [
    {"id": 1, "n_reads": <int>, "configs": [
      {"level": 0, "answer_path": "<path or null>"},
      {"level": 1, "answer_path": "<path or null>"},
      {"level": 2, "answer_path": "<path or null>"}
    ]},
    ...same for q2..q17, then q19, q20 (skip q18)...
  ]
}

Do NOT grade yourself; do NOT mention what file "should" be the answer.

Begin.
```

---

## Workflow

For each backend in the list above:

1. Substitute `<BACKEND>` everywhere in the prompt above (2 places).
2. Open new Claude Code chat.
3. Paste the modified prompt.
4. Wait for JSON output (~24 tool calls, ~2 minutes).
5. Save the JSON to `fresh_results_acropolis_<BACKEND>.json` in
   `context-transfer-engine/test/unit/`.

When all 7 are done, paste me a confirmation ("done with 7") and I'll:
- Grade each one with the multi-answer grader (q19 accepts both `.cc` and `.h`)
- Mine each session JSONL for real billed tokens
- Render a per-backend comparison figure showing accuracy + tokens with
  baseline as reference

---

## Why q18 was removed

q18 ("Locate the file that defines the kCtePoolName and kCtePoolId
constants") names exact identifier strings — it is a pure lexical lookup
that grep handles trivially and that dense embeddings cannot represent
faithfully (BPE shreds rare identifiers; cosine to a paraphrased prose
summary is near-random). Including it in a semantic-search benchmark
biases the comparison toward grep on its home turf. It remains a valid
"lexical robustness" stress test that hybrid backends should handle, but
it is not part of the headline accuracy metric.

---

## Expected outcomes by backend paradigm

| Paradigm | Backends | Expected L2 accuracy |
|---|---|---|
| Pure lexical (BM25-class) | `bm25`, `elasticsearch-kw`, `neo4j-kw` | high on identifier-laden queries, mid on paraphrase |
| Pure dense (cosine) | `elasticsearch-vec`, `qdrant` | high on paraphrase, low on rare-term queries |
| Hybrid (RRF lexical + dense) | `elasticsearch-rrf`, `neo4j-rrf` | best of both — should land closest to baseline |

If hybrid backends land at ≥ 17/19, that's the headline result for the
paper. If pure-dense underperforms while hybrid recovers, that confirms
the well-known finding (Anthropic Contextual Retrieval, 2024) that dense
retrieval on code requires a lexical complement.
