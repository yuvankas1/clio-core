#!/usr/bin/env python3
"""Benchmark Chroma's context-1 (chromadb/context-1) on the 18 Acropolis queries.

Comparison framing for the paper: end-to-end pipeline vs. end-to-end pipeline.
Each system (Acropolis backends, Chroma context-1, ...) gets the raw clio-core
repo and must answer the 18 queries. Chroma's L2 summary cache is NOT shared
(that's Acropolis's own indexing artifact).

This script:
  1. Parses the 18 queries from PROTOCOL_ACROPOLIS.md
  2. Indexes clio-core into a Chroma collection (chunked content + cross-encoder)
     -- reuses .chroma_context1_store/ if already built with same chunk count
  3. Loads chromadb/context-1 (gpt-oss-20b BF16 fine-tune) via transformers
  4. For each query, runs a ReAct loop with 4 tools:
       search_corpus(query, k)        - hybrid BM25+dense+RRF+rerank
       grep_corpus(pattern, k)        - regex/substring search across files
       read_document(path)            - first 4 KB of a file
       prune_chunks(keep_paths)       - no-op stub
     Model output is parsed as either:
       {"tool_call": {"name": "...", "args": {...}}}
       {"final_answer": "<relative-path-from-repo-root>"}
  5. Grades each answer via plot_fresh_results.EXPECTED + hit()
  6. Writes fresh_results_context1.json next to this script.
"""

import argparse
import glob
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]               # /u/rpawar/clio-core
PROTOCOL_MD = HERE / "PROTOCOL_ACROPOLIS.md"
CHROMA_DIR = HERE / ".chroma_context1_store"
COLLECTION = "clio_core_context1"
OUT_JSON = HERE / "fresh_results_context1.json"

# Indexing config (matches run_chroma_context1.py)
CHUNK_TOKENS = 512
CHUNK_OVERLAP = 64
RETRIEVE_TOP = 50            # before fusion+rerank
RRF_K = 60
TOP_OUT_DEFAULT = 5

# Agent loop config
MAX_TURNS = 4
GEN_MAX_NEW_TOKENS = 512
READ_BYTES = 4096

SKIP_DIRS = {".git", ".cache", "node_modules", "__pycache__", ".chroma_context1_store"}
SKIP_DIR_PREFIXES = ("build", "build_", ".venv", "venv")
INDEX_EXTS = {
    ".cc", ".cpp", ".cxx", ".c", ".h", ".hpp", ".hh", ".cuh", ".cu",
    ".py", ".pyx",
    ".yaml", ".yml", ".toml", ".ini",
    ".json", ".md", ".rst", ".txt",
    ".sh", ".bash",
    ".cmake",
}
INDEX_BASENAMES = {"CMakeLists.txt", "Dockerfile", "Makefile"}
MAX_FILE_BYTES = 512 * 1024

# Benchmark-scaffolding files quote queries/expected answers verbatim -- exclude
# from the corpus so we test retrieval, not memorization of the answer key.
EXCLUDE_RELPATHS = {
    "context-transfer-engine/test/unit/PROTOCOL_ACROPOLIS.md",
    "context-transfer-engine/test/unit/PROTOCOL_BASELINE.md",
    "context-transfer-engine/test/unit/bench_repo_scan.cc",
    "context-transfer-engine/test/unit/plot_fresh_results.py",
    "context-transfer-engine/test/unit/run_chroma_context1.py",
    "context-transfer-engine/test/unit/grade_chroma_context1.py",
    "context-transfer-engine/test/unit/chroma_context1_bench.py",
    "context-transfer-engine/test/unit/acropolis_mcp_server.py",
    "context-transfer-engine/test/unit/acropolis_mcp_README.md",
    "context-transfer-engine/test/unit/resummarize_pilot.py",
    "context-transfer-engine/test/unit/mine_session_tokens.py",
    "context-transfer-engine/test/unit/fresh_results_acropolis.json",
    "context-transfer-engine/test/unit/fresh_results_baseline.json",
    "context-transfer-engine/test/unit/fresh_results_chroma_context1.json",
    "context-transfer-engine/test/unit/fresh_results_context1.json",
    "context-transfer-engine/test/unit/run_chroma_context1.log",
}

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


# ----- Query parsing from PROTOCOL_ACROPOLIS.md ------------------------------

QUERY_LINE_RE = re.compile(r'^\s+q(\d+):\s+"(.+)"\s*$')


def parse_queries() -> list[tuple[int, str]]:
    """Parse the 18 queries from PROTOCOL_ACROPOLIS.md.

    The protocol numbers queries 1..17 then 19, 20 (q18 is intentionally absent).
    """
    text = PROTOCOL_MD.read_text()
    queries: list[tuple[int, str]] = []
    seen: set[int] = set()
    for line in text.splitlines():
        m = QUERY_LINE_RE.match(line)
        if m:
            qid = int(m.group(1))
            if qid in seen:
                continue
            seen.add(qid)
            queries.append((qid, m.group(2)))
    return queries


# ----- File walking + chunking ----------------------------------------------

def should_index(path: Path) -> bool:
    if path.name in INDEX_BASENAMES:
        return True
    return path.suffix.lower() in INDEX_EXTS


def iter_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames
            if d not in SKIP_DIRS and not any(d.startswith(p) for p in SKIP_DIR_PREFIXES)
        ]
        for name in filenames:
            p = Path(dirpath) / name
            if not should_index(p):
                continue
            try:
                rel = str(p.relative_to(root))
            except ValueError:
                continue
            if rel in EXCLUDE_RELPATHS:
                continue
            try:
                if p.stat().st_size > MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            yield p


def read_text_safe(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def chunk_text(text: str, chunk_tokens=CHUNK_TOKENS, overlap=CHUNK_OVERLAP):
    words = text.split()
    if not words:
        return []
    out = []
    step = max(1, chunk_tokens - overlap)
    for start in range(0, len(words), step):
        end = min(start + chunk_tokens, len(words))
        out.append((start, end, " ".join(words[start:end])))
        if end >= len(words):
            break
    return out


# ----- BM25 (matches Acropolis MCP server's variant) ------------------------

def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


class BM25:
    def __init__(self, texts: list[str], k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.tok = [tokenize(t) for t in texts]
        self.dl = [len(t) for t in self.tok]
        self.N = len(texts)
        self.avgdl = (sum(self.dl) / self.N) if self.N else 0.0
        self.df: Counter[str] = Counter()
        for toks in self.tok:
            for term in set(toks):
                self.df[term] += 1
        self.tf = [Counter(t) for t in self.tok]

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        q_terms = tokenize(query)
        scored = []
        for i in range(self.N):
            if self.dl[i] == 0:
                continue
            s = 0.0
            for q in q_terms:
                if q not in self.df:
                    continue
                idf = math.log((self.N - self.df[q] + 0.5) / (self.df[q] + 0.5) + 1)
                tf = self.tf[i][q]
                denom = tf + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl)
                s += idf * (tf * (self.k1 + 1)) / denom
            if s > 0:
                scored.append((i, s))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]


def rrf_fuse(rankings: list[list[int]], k=RRF_K) -> list[tuple[int, float]]:
    score: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            score[doc_id] += 1.0 / (k + rank + 1)
    return sorted(score.items(), key=lambda x: x[1], reverse=True)


# ----- Indexing + tools ------------------------------------------------------

class Corpus:
    """Walks clio-core, chunks files, builds (or reuses) a Chroma collection
    + BM25 over the same chunks, and loads a cross-encoder for reranking."""

    def __init__(self, root: Path):
        import chromadb
        from chromadb.utils import embedding_functions
        from sentence_transformers import CrossEncoder

        print(f"[corpus] walking {root}", flush=True)
        self.files = list(iter_files(root))
        print(f"[corpus] {len(self.files)} files", flush=True)
        self.chunks: list[str] = []
        self.metas: list[dict] = []
        for fp in self.files:
            text = read_text_safe(fp)
            if not text:
                continue
            rel = str(fp.relative_to(root))
            for start, end, body in chunk_text(text):
                self.chunks.append(body)
                self.metas.append({"path": rel, "start": start, "end": end})
        self.n = len(self.chunks)
        print(f"[corpus] {self.n} chunks", flush=True)

        print(f"[corpus] opening Chroma persistent client at {CHROMA_DIR}", flush=True)
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.embedder = embedding_functions.DefaultEmbeddingFunction()
        reuse = False
        try:
            existing = self.client.get_collection(COLLECTION, embedding_function=self.embedder)
            if existing.count() == self.n:
                self.coll = existing
                reuse = True
                print(f"[corpus] reusing collection ({existing.count()} chunks)", flush=True)
        except Exception:
            pass
        if not reuse:
            try:
                self.client.delete_collection(COLLECTION)
            except Exception:
                pass
            self.coll = self.client.create_collection(
                COLLECTION, embedding_function=self.embedder,
                metadata={"hnsw:space": "cosine"},
            )
            BATCH = 256
            ids = [f"c{i}" for i in range(self.n)]
            for i in range(0, self.n, BATCH):
                self.coll.add(
                    ids=ids[i:i+BATCH],
                    documents=self.chunks[i:i+BATCH],
                    metadatas=self.metas[i:i+BATCH],
                )
                print(f"[corpus] embedded {min(i+BATCH, self.n)}/{self.n}", flush=True)

        print("[corpus] building BM25", flush=True)
        self.bm25 = BM25(self.chunks)
        print("[corpus] loading cross-encoder", flush=True)
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        print("[corpus] ready", flush=True)

        self.root = root
        self.seen_chunks: set[int] = set()  # for search_corpus dedup

    # ---- Tool: search_corpus ----
    def search_corpus(self, query: str, k: int = TOP_OUT_DEFAULT) -> list[dict]:
        dense = self.coll.query(query_texts=[query], n_results=RETRIEVE_TOP)
        dense_ids = [int(x[1:]) for x in dense["ids"][0]]
        bm25_hits = self.bm25.search(query, RETRIEVE_TOP)
        bm25_ids = [i for i, _ in bm25_hits]
        fused = rrf_fuse([dense_ids, bm25_ids])
        pool = [d for d, _ in fused if d not in self.seen_chunks][:RETRIEVE_TOP]
        if not pool:
            return []
        pairs = [(query, self.chunks[i]) for i in pool]
        scores = self.reranker.predict(pairs).tolist()
        reranked = sorted(zip(pool, scores), key=lambda x: x[1], reverse=True)[:k]
        out = []
        for cid, sc in reranked:
            self.seen_chunks.add(cid)
            out.append({
                "id": int(cid),
                "path": self.metas[cid]["path"],
                "score": round(float(sc), 4),
                "snippet": self.chunks[cid][:600],
            })
        return out

    # ---- Tool: grep_corpus ----
    def grep_corpus(self, pattern: str, k: int = TOP_OUT_DEFAULT) -> list[dict]:
        try:
            rx = re.compile(pattern, re.IGNORECASE)
        except re.error:
            rx = re.compile(re.escape(pattern), re.IGNORECASE)
        out: list[dict] = []
        seen_paths: set[str] = set()
        for fp in self.files:
            if len(out) >= k:
                break
            text = read_text_safe(fp)
            if not text:
                continue
            for lineno, line in enumerate(text.splitlines(), 1):
                if rx.search(line):
                    rel = str(fp.relative_to(self.root))
                    if rel in seen_paths:
                        continue
                    seen_paths.add(rel)
                    out.append({"path": rel, "line": lineno, "text": line[:300]})
                    break
        return out

    # ---- Tool: read_document ----
    def read_document(self, path: str) -> str:
        # Path is relative-to-repo-root; defend against escape.
        rel = path.lstrip("/")
        full = (self.root / rel).resolve()
        try:
            full.relative_to(self.root.resolve())
        except ValueError:
            return f"<error: path {path!r} outside repo>"
        if not full.is_file():
            return f"<error: not a file: {path!r}>"
        try:
            with open(full, "r", encoding="utf-8", errors="replace") as f:
                return f.read(READ_BYTES)
        except OSError as e:
            return f"<error: {e}>"

    # ---- Tool: prune_chunks (stub) ----
    def prune_chunks(self, keep_paths: list[str]) -> dict:
        return {"ok": True, "note": "prune is a no-op stub in this harness"}


# ----- ReAct agent loop ------------------------------------------------------

DEVELOPER_INSTRUCTIONS = """\
You are a code-search agent. Given one natural-language query about the
clio-core codebase, identify the single file path (relative to repo root)
that best answers it.

# Required workflow

1. FIRST, call the `search_corpus` tool. Do NOT emit a final answer before
   you have called at least one tool.
2. Inspect the returned chunks. **A header file (.h/.hpp) is a perfectly
   valid answer** -- many backends in this codebase are implemented entirely
   inline in headers (no .cc file exists). If the top hit is a header whose
   snippet shows real class bodies or function implementations, that IS the
   implementation file. Do not keep hunting for a `.cc` file that may not
   exist.
3. **Commit decisively.** If 2-3 searches return the same top candidate, or
   the very first search returns a clear winner, COMMIT IT. Do not loop
   indefinitely refining queries.
4. When you have identified the answer, emit your final-channel message
   containing EXACTLY this JSON object on a single line:

       {"final_answer": "<relative-path-from-repo-root>"}

# Tools

  search_corpus(query)        -- hybrid BM25+dense+rerank; returns chunks
                                 with paths and snippets. Never re-returns a
                                 chunk it has already returned.
  grep_corpus(pattern)        -- regex/substring across files; for exact
                                 identifiers, string constants, filenames.
  read_document(path)         -- first 4 KB of a file; use to verify.
  prune_chunks(keep_paths)    -- no-op stub; ignore.

# Constraints

- Maximum 6 turns. Commit early if you have a confident candidate.
- The path you commit MUST be a path that appeared in a tool result.
- Never emit anything on the final channel except the JSON answer.
- If you have searched 3 times with no clearly-better candidate, commit
  the strongest path you have seen so far rather than continuing to search."""


TOOLS_OPENAI = [
    {
        "type": "function",
        "function": {
            "name": "search_corpus",
            "description": "Hybrid BM25 + dense vector search with cross-encoder reranking over the indexed code corpus. Returns up to k chunks not previously returned in this conversation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural-language search query."},
                    "k": {"type": "integer", "description": "Max number of chunks to return.", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep_corpus",
            "description": "Regex / substring search across raw file contents. Returns up to k files where the first matching line is shown. Use for exact identifiers, string constants, or specific filenames.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex or substring."},
                    "k": {"type": "integer", "description": "Max number of files.", "default": 5},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_document",
            "description": "Read the first 4 KB of the file at the given path (relative to repo root). Use to verify a candidate file is the right one.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to repo root."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "prune_chunks",
            "description": "No-op stub in this harness; ignore.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keep_paths": {"type": "array", "items": {"type": "string"},
                                   "description": "Paths to keep; everything else is pruned."},
                },
                "required": ["keep_paths"],
            },
        },
    },
]


# Harmony tool-call output looks like (with special tokens):
#   <|start|>assistant to=functions.NAME<|channel|>commentary json<|message|>{args}<|call|>
# Analysis CoT:
#   <|channel|>analysis<|message|>...<|end|>
# Final answer:
#   <|channel|>final<|message|>...<|return|>
# After the generation prompt `<|start|>assistant\n`, the FIRST segment skips
# the leading `<|start|>assistant`. The regex makes that prefix optional.
_HARMONY_SEG_RE = re.compile(
    r"(?:<\|start\|>assistant)?"
    r"(?:\s+to=functions\.([A-Za-z_][A-Za-z0-9_]*))?"
    r"<\|channel\|>([a-z]+)"
    # Optional channel suffix: space, optional <|constrain|>, then a content
    # type word like 'json'. Real outputs include
    #   'commentary json'
    #   'commentary <|constrain|>json'
    r"(?:\s*(?:<\|constrain\|>)?\s*[a-z]+)?"
    r"<\|message\|>"
    r"(.*?)"
    # vLLM strips the trailing <|call|>/<|return|> stop token from output
    # text. Accept \Z (absolute end of string) so a truncated/terminator-less
    # final segment still parses.
    r"(<\|end\|>|<\|call\|>|<\|return\|>|\Z)",
    re.DOTALL,
)


def parse_harmony(text: str) -> list[dict]:
    """Walk harmony output into segments.

    Returns dicts: {kind: 'tool_call'|'text', channel, name?, body}.
    Stops at the first <|return|> terminator.
    """
    out: list[dict] = []
    for m in _HARMONY_SEG_RE.finditer(text):
        tool_name = m.group(1)
        channel = m.group(2)
        body = m.group(3).strip()
        terminator = m.group(4)
        if tool_name:
            out.append({"kind": "tool_call", "name": tool_name, "channel": channel, "body": body})
        else:
            out.append({"kind": "text", "channel": channel, "body": body})
        if terminator == "<|return|>":
            break
    return out


_FINAL_ANSWER_JSON_RE = re.compile(r'"final_answer"\s*:\s*"([^"]+)"')
_PATH_LIKE_RE = re.compile(
    r'([A-Za-z0-9_./-]+\.(?:cc|cpp|cxx|c|h|hpp|hh|cuh|cu|py|yaml|yml|json|md|sh|cmake|txt))'
)


def extract_path(final_body: str) -> str | None:
    """Best-effort path extraction from the model's final channel content."""
    if not final_body:
        return None
    m = _FINAL_ANSWER_JSON_RE.search(final_body)
    if m:
        return m.group(1)
    m = _PATH_LIKE_RE.search(final_body)
    if m:
        return m.group(1)
    # As a last resort, trimmed body if it looks like a single path.
    stripped = final_body.strip().strip('"').strip()
    if "/" in stripped and len(stripped.splitlines()) == 1 and len(stripped) < 250:
        return stripped
    return None


def run_agent(llm, tokenizer, sampling_params, corpus: Corpus,
              qid: int, query: str) -> dict:
    """Harmony-native agent loop for one query, served by vLLM.

    Model: chromadb/context-1 (gpt-oss-20b BF16 fine-tune), trained on the
    harmony format: analysis CoT / commentary tool-call channel / final
    answer channel.

    Loop: render messages+tools through the model's chat template into a
    harmony-formatted prompt, generate with vLLM, parse the raw output
    (special tokens kept) into harmony segments. Execute the first tool call
    in each turn and append the result as a `role: tool` message; the chat
    template renders that as a harmony tool-result back to the model.
    """
    messages: list[dict] = [
        {"role": "developer", "content": DEVELOPER_INSTRUCTIONS},
        {"role": "user", "content": query},
    ]
    trace: list[dict] = []
    answer_path: str | None = None
    n_tool_calls = 0
    n_turns = 0
    t0 = time.time()
    corpus.seen_chunks.clear()

    def gen(messages_):
        # Use vLLM's chat() API directly so it applies the model's chat
        # template with the correct tokenizer settings (special tokens like
        # <|start|>/<|channel|> map to single IDs, no double-encoding).
        # Pre-rendering to string + LLM.generate() produced random-token
        # gibberish — vLLM was re-tokenizing the harmony markers as text.
        outputs = llm.chat(
            messages=messages_,
            sampling_params=sampling_params,
            tools=TOOLS_OPENAI,
            chat_template_kwargs={"reasoning_effort": "high"},
            use_tqdm=False,
        )
        prompt_tok_count = len(outputs[0].prompt_token_ids or [])
        raw = outputs[0].outputs[0].text
        return prompt_tok_count, raw

    for turn in range(MAX_TURNS):
        n_turns = turn + 1
        prompt_len, raw = gen(messages)
        segments = parse_harmony(raw)
        trace.append({
            "turn": turn,
            "prompt_tokens": int(prompt_len),
            "generated_chars": len(raw),
            "raw_first_2000": raw[:2000],
            "segments": [
                {"kind": s["kind"], "channel": s.get("channel"),
                 "name": s.get("name"), "body_first_500": s["body"][:500]}
                for s in segments
            ],
        })

        tool_calls = [s for s in segments if s["kind"] == "tool_call"]
        analysis_segs = [s for s in segments if s["kind"] == "text" and s["channel"] == "analysis"]
        final_segs = [s for s in segments if s["kind"] == "text" and s["channel"] == "final"]

        if tool_calls:
            # Append assistant message with the (first) tool_call. Multiple
            # tool calls in one turn would each go in their own message; the
            # chat template states "max 1 tool call per message". Emit the
            # first; remaining tool calls will be re-emitted next turn if the
            # model wants to.
            tc = tool_calls[0]
            try:
                args = json.loads(tc["body"]) if tc["body"] else {}
            except json.JSONDecodeError:
                args = {}
            analysis_text = "\n".join(s["body"] for s in analysis_segs) or None
            messages.append({
                "role": "assistant",
                "content": analysis_text,
                "tool_calls": [{
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": args},
                }],
            })
            n_tool_calls += 1
            name = tc["name"]
            if name == "search_corpus":
                result = corpus.search_corpus(args.get("query", ""),
                                              int(args.get("k", TOP_OUT_DEFAULT)))
            elif name == "grep_corpus":
                result = corpus.grep_corpus(args.get("pattern", ""),
                                            int(args.get("k", TOP_OUT_DEFAULT)))
            elif name == "read_document":
                result = corpus.read_document(args.get("path", ""))
            elif name == "prune_chunks":
                result = corpus.prune_chunks(list(args.get("keep_paths", [])))
            else:
                result = {"error": f"unknown tool {name!r}"}
            messages.append({"role": "tool", "content": json.dumps(result)})
            continue

        if final_segs:
            final_text = "\n".join(s["body"] for s in final_segs)
            answer_path = extract_path(final_text)
            messages.append({"role": "assistant", "content": final_text})
            break

        # No tool call, no final channel -- only analysis (or nothing parseable).
        # Nudge the model.
        if analysis_segs:
            nudge = ("Continue: either call a tool on the commentary channel, or "
                     "emit your final answer on the final channel as "
                     '{"final_answer": "<path>"}.')
            messages.append({"role": "assistant",
                             "content": "\n".join(s["body"] for s in analysis_segs)})
            messages.append({"role": "user", "content": nudge})
            continue

        # Truly nothing — bail.
        break

    # If we exhausted MAX_TURNS without a committed answer, do one final
    # "commit-forcing" turn: drop the tools, instruct the model to emit ONLY
    # the JSON final_answer. The model often gets stuck doing extra searches
    # after it already has the right candidate (it knows the answer in its
    # analysis CoT but never moves to the final channel).
    if answer_path is None:
        force_text = (
            "STOP. You must now commit a final answer. Look at the paths you "
            "have already seen in your tool results. Pick the single best "
            "match for the query and reply with ONLY the JSON object on the "
            'final channel: {"final_answer": "<relative-path-from-repo-root>"} '
            "-- no analysis, no preamble, no further tool calls. Just the JSON."
        )
        messages.append({"role": "user", "content": force_text})
        try:
            outputs = llm.chat(
                messages=messages,
                sampling_params=sampling_params,
                # no tools= -> the model can't call tools, only emit final
                chat_template_kwargs={"reasoning_effort": "low"},
                use_tqdm=False,
            )
            raw_force = outputs[0].outputs[0].text
        except Exception as e:
            raw_force = f"<forcing-turn error: {e}>"
        trace.append({
            "turn": "force",
            "prompt_tokens": int(len(outputs[0].prompt_token_ids or []) if "outputs" in dir() else 0),
            "generated_chars": len(raw_force),
            "raw_first_2000": raw_force[:2000],
            "segments": [],
        })
        # Look at the model's final-channel and analysis CoT for a path/JSON.
        seg_for = parse_harmony(raw_force)
        for s in seg_for:
            if s["kind"] == "text":
                p = extract_path(s["body"])
                if p:
                    answer_path = p
                    break
        # Last resort: scan the entire trace's analysis CoT for a path.
        if answer_path is None:
            for t in trace:
                for s in t.get("segments", []):
                    p = extract_path(s.get("body_first_500", ""))
                    if p:
                        answer_path = p
                        break
                if answer_path:
                    break

    return {
        "id": qid,
        "query": query,
        "answer_path": answer_path,
        "n_turns": n_turns,
        "tool_calls": n_tool_calls,
        "elapsed_s": round(time.time() - t0, 2),
        "trace_summary": trace,
    }


# ----- Grading via plot_fresh_results -----------------------------------------

def load_grader():
    sys.path.insert(0, str(HERE))
    from plot_fresh_results import EXPECTED, hit, QID_TO_POS  # noqa: F401
    return EXPECTED, hit


# ----- Main -----------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="chromadb/context-1",
                    help="HF repo id or local path to the model")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(OUT_JSON))
    args = ap.parse_args()

    print(f"=== Acropolis-vs-Chroma benchmark on Chroma context-1 ===", flush=True)
    print(f"repo root: {REPO_ROOT}", flush=True)

    queries = parse_queries()
    print(f"parsed {len(queries)} queries from PROTOCOL_ACROPOLIS.md", flush=True)

    EXPECTED, hit = load_grader()
    # Keep only queries the existing grader scores -- q17 is intentionally
    # excluded ("deferred-release lives past bench's first-4KB truncation"),
    # so EXPECTED has 18 keys: q1-q16, q19, q20.
    pre_count = len(queries)
    queries = [(qid, q) for qid, q in queries if qid in EXPECTED]
    print(f"  using {len(queries)} graded queries (dropped {pre_count - len(queries)} not in EXPECTED)", flush=True)
    for qid, q in queries:
        print(f"    q{qid}: {q[:80]}", flush=True)

    if args.limit:
        queries = queries[:args.limit]

    corpus = Corpus(REPO_ROOT)

    print(f"[model] loading {args.model} via vLLM", flush=True)
    t0 = time.time()
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # vLLM handles memory-efficient inference (PagedAttention + FlashAttention
    # backend for gpt-oss). Cap max_model_len so we don't preallocate KV cache
    # for the model's 131k native context.
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=32768,
        gpu_memory_utilization=0.85,
        trust_remote_code=False,
        # enforce_eager=True disables vLLM's torch.compile + CUDA-graph
        # capture, which avoids the FlashInfer fused-MoE JIT ninja build that
        # OOM-killed in earlier runs. Inference is slower but reliable, and
        # we still use FlashAttention for memory-efficient attention.
        enforce_eager=True,
        # The default FlashInfer CUTLASS Unquantized MoE backend produced
        # complete-gibberish outputs on this GH200 + gpt-oss-20b combo
        # (raw and chat-template prompts both yielded random tokens). The
        # TRITON backend works correctly — confirmed by debug_vllm.py.
        kernel_config={"moe_backend": "triton"},
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=GEN_MAX_NEW_TOKENS,
        skip_special_tokens=False,
    )
    print(f"[model] loaded in {time.time()-t0:.1f}s", flush=True)

    results = []
    for qid, q in queries:
        print(f"\n=== q{qid}: {q} ===", flush=True)
        r = run_agent(llm, tokenizer, sampling_params, corpus, qid, q)
        # Inline grade so we get live feedback
        if qid in EXPECTED:
            r["hit"] = hit(r["answer_path"], EXPECTED[qid])
            r["expected"] = EXPECTED[qid]
        else:
            r["hit"] = False
            r["expected"] = None
        results.append(r)
        print(f"q{qid}: {r['n_turns']} turns, hit={'Y' if r['hit'] else 'N'}, "
              f"path={r['answer_path']!r}", flush=True)

    n_hit = sum(1 for r in results if r["hit"])
    print(f"\n=== Context-1 score: {n_hit}/{len(results)} ===", flush=True)

    # Find newest summary cache file (informational only; not used by Chroma)
    cache_files = sorted(
        HERE.glob("acropolis_summary_cache_*.json"),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    summary_cache = cache_files[0].name if cache_files else None

    out = {
        "agent": "chromadb/context-1",
        "arm": "context1_minimal_harness",
        "model_id": args.model,
        "summary_cache": summary_cache,
        "total_queries": len(results),
        "score": n_hit,
        "system": {
            "corpus": "raw clio-core (chunked file contents) -- NOT Acropolis's L2 summaries",
            "n_files": len(corpus.files),
            "n_chunks": corpus.n,
            "embedder": "chroma default ONNX MiniLM-L6-v2 (384-d)",
            "lexical": "BM25 pure-python (k1=1.5 b=0.75)",
            "fusion": f"RRF (k={RRF_K})",
            "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "tools": ["search_corpus", "grep_corpus", "read_document", "prune_chunks(stub)"],
            "max_turns": MAX_TURNS,
        },
        "results": results,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}", flush=True)

    print("\n=== Comparison vs Acropolis ===", flush=True)
    print("  Acropolis oracle:    18/18")
    print("  Acropolis routed:    18/18")
    print("  Acropolis baseline:  17/18")
    print(f"  Chroma context-1:    {n_hit}/{len(results)}")
    print("\nMisses:")
    for r in results:
        if not r["hit"]:
            print(f"  q{r['id']}: expected {r['expected']} got {r['answer_path']!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
