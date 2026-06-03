#!/usr/bin/env python3
"""Acropolis L2 indexing ablation: sweep (summary_model x n_sentences) and
score against the 18-query benchmark on bm25, qdrant, and elasticsearch-rrf.

Pipeline per cell (model, n_sentences):
  1. Walk clio-core (same EXCLUDE_RELPATHS as the chroma run).
  2. For each file, call Ollama with a prompt parametrized by n_sentences,
     producing a path -> summary cache.
  3. Index summaries:
       - BM25: pure Python (mirrors acropolis_mcp_server.BM25)
       - Qdrant: in-process client, nomic-embed-text via Ollama
       - Elasticsearch: HTTP, kw + vec + RRF
  4. Run the 18 queries through each backend, take top-1, grade via
     plot_fresh_results.EXPECTED.

Per-cell record:
  cell_id, model, n_sentences, n_files, total_summary_bytes, mean_summary_bytes,
  summarization_total_s, summarization_per_file_s,
  per_backend: {accuracy/N, hits, misses, avg_query_s}

Output: fresh_results_ablation.json + plotted figures.
"""

import argparse
import glob
import json
import math
import os
import re
import sys
import time
import urllib.request
import urllib.error
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]                  # /u/rpawar/clio-core
PROTOCOL_MD = HERE / "PROTOCOL_ACROPOLIS.md"
OUT_DIR = Path(os.environ.get("ACROPOLIS_ABLATION_OUT",
                              "/u/rpawar/acropolis_ablation"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

OLLAMA_URL = os.environ.get("ACROPOLIS_OLLAMA_URL", "http://localhost:11434")
ES_URL     = os.environ.get("ACROPOLIS_ES_URL",     "http://localhost:9200")
EMBED_MODEL = os.environ.get("ACROPOLIS_EMBED_MODEL", "nomic-embed-text")

INDEX_EXTS = {
    ".cc", ".cpp", ".cxx", ".c", ".h", ".hpp", ".hh", ".cuh", ".cu",
    ".py", ".pyx", ".yaml", ".yml", ".toml", ".ini",
    ".json", ".md", ".rst", ".txt", ".sh", ".bash", ".cmake",
}
INDEX_BASENAMES = {"CMakeLists.txt", "Dockerfile", "Makefile"}
MAX_FILE_BYTES = 512 * 1024
# CAE C++ summary_operator caps file content at 4096 bytes (raw cap, NOT
# head+tail split). Match exactly so our HTTP request body is byte-identical
# to what bench_repo_scan.cc would send.
MAX_FILE_CHARS = 4096

SKIP_DIRS = {".git", ".cache", "node_modules", "__pycache__",
             ".chroma_context1_store"}
SKIP_DIR_PREFIXES = ("build", "build_", ".venv", "venv")

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
    "context-transfer-engine/test/unit/acropolis_ablation.py",
    "context-transfer-engine/test/unit/plot_acropolis_ablation.py",
    "context-transfer-engine/test/unit/plot_chroma_vs_acropolis.py",
    "context-transfer-engine/test/unit/plot_chroma_tokens.py",
    "context-transfer-engine/test/unit/plot_token_3way.py",
    "context-transfer-engine/test/unit/fresh_results_acropolis.json",
    "context-transfer-engine/test/unit/fresh_results_baseline.json",
    "context-transfer-engine/test/unit/fresh_results_chroma_context1.json",
    "context-transfer-engine/test/unit/fresh_results_context1.json",
    "context-transfer-engine/test/unit/fresh_results_ablation.json",
    "context-transfer-engine/test/unit/run_chroma_context1.log",
}

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
QUERY_LINE_RE = re.compile(r'^\s+q(\d+):\s+"(.+)"\s*$')


# ----- Files + queries -------------------------------------------------------

def should_index(p: Path) -> bool:
    if p.name in INDEX_BASENAMES:
        return True
    return p.suffix.lower() in INDEX_EXTS


def iter_files(root: Path):
    for dp, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs
                   if d not in SKIP_DIRS and not any(d.startswith(p) for p in SKIP_DIR_PREFIXES)]
        for name in files:
            p = Path(dp) / name
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
            yield rel, p


def read_text(p: Path) -> str:
    try:
        with open(p, encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def parse_queries() -> list[tuple[int, str]]:
    out = []
    seen = set()
    for line in PROTOCOL_MD.read_text().splitlines():
        m = QUERY_LINE_RE.match(line)
        if m:
            qid = int(m.group(1))
            if qid in seen:
                continue
            seen.add(qid)
            out.append((qid, m.group(2)))
    return out


def load_grader():
    sys.path.insert(0, str(HERE))
    from plot_fresh_results import EXPECTED, hit
    return EXPECTED, hit


# ----- Ollama HTTP -----------------------------------------------------------

def ollama_chat(model: str, system: str, user: str, num_predict: int = 400) -> str:
    """CAE-equivalent HTTP request to Ollama's OpenAI-compatible endpoint.

    Matches bench_repo_scan.cc + summary_operator.cc byte-for-byte:
      - URL:         {base}/chat/completions  (NOT /api/chat)
      - body:        {model, messages, max_tokens, temperature: 0.0}
      - response:    choices[0].message.content
    At temperature 0.0 the model is deterministic; with identical
    (model, prompt, file_content, max_tokens) the output is bit-identical to
    what the C++ CAE pipeline would produce.
    """
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "max_tokens": num_predict,
        "temperature": 0.0,
    }
    req = urllib.request.Request(
        f"{OLLAMA_URL}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        d = json.loads(r.read().decode())
    choices = d.get("choices") or []
    if not choices:
        return ""
    msg = choices[0].get("message") or {}
    return (msg.get("content") or "").strip()


def ollama_embed(text: str, model: str = EMBED_MODEL,
                 timeout: int = 300, retries: int = 2) -> list[float] | None:
    body = {"model": model, "input": text}
    for attempt in range(retries + 1):
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/embed",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.loads(r.read().decode())
            emb = d.get("embeddings") or []
            return emb[0] if emb else None
        except Exception as e:           # TimeoutError, URLError, etc.
            if attempt == retries:
                print(f"  ollama_embed failed after {retries+1} attempts: {e}", flush=True)
                return None
            time.sleep(2 ** attempt)


def warmup_embedder():
    """Force Ollama to load nomic-embed-text once so the first real call
    in a cell doesn't time out while the model is being swapped to GPU."""
    print("[warmup] loading nomic-embed-text into GPU memory ...", flush=True)
    t0 = time.time()
    v = ollama_embed("warmup", timeout=600, retries=0)
    print(f"[warmup] embedder ready in {time.time()-t0:.0f}s "
          f"(dim={len(v) if v else 'FAIL'})", flush=True)


# ----- Summary prompt --------------------------------------------------------

def build_system_prompt(n_sentences: int) -> str:
    """Generalisation of search_optimized_summary_v2.txt to N sentences."""
    word_lo = max(20, n_sentences * 18)
    word_hi = n_sentences * 32
    return (
        f"You write {n_sentences}-sentence summaries of source files for semantic search. "
        f"Developers will query the codebase in natural language and expect this "
        f"file to surface when their query touches its purpose, key symbols, or "
        f"techniques.\n\n"
        f"Rules:\n"
        f"- Exactly {n_sentences} sentences, ~{word_lo}-{word_hi} words total.\n"
        f"- Sentence 1: state what the file does and name its main symbol "
        f"verbatim (class/function/constant -- in backticks).\n"
        f"- Remaining sentences: pack in distinctive terms -- algorithm names, "
        f"design patterns, technical jargon from comments (e.g., 'FlexGen', "
        f"'Reciprocal Rank Fusion', 'double buffering', 'deferred release'), "
        f"notable constants, and domain-specific terminology.\n"
        f"- Always include both abbreviations AND expanded forms when present "
        f"(e.g., 'Reciprocal Rank Fusion (RRF)').\n"
        f"- Avoid generic phrases like 'implements a backend' or 'manages memory'.\n"
        f"- Return ONLY the summary, no preamble."
    )


def build_user_msg(rel: str, content: str) -> str:
    if len(content) > MAX_FILE_CHARS:
        head = content[: MAX_FILE_CHARS // 2]
        tail = content[-MAX_FILE_CHARS // 2:]
        content = head + "\n\n... [truncated] ...\n\n" + tail
    return f"FILE PATH: {rel}\n\nFILE CONTENT:\n```\n{content}\n```"


# ----- BM25 over (path + summary) -- mirrors acropolis_mcp_server -----------

def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


class BM25:
    def __init__(self, docs: list[tuple[str, str]], k1=1.5, b=0.75):
        self.docs = docs
        self.k1, self.b = k1, b
        self.tok = [tokenize(f"{p}\n{s}") for p, s in docs]
        self.dl = [len(t) for t in self.tok]
        self.N = len(docs)
        self.avgdl = sum(self.dl) / self.N if self.N else 0.0
        self.df = Counter()
        for toks in self.tok:
            for term in set(toks):
                self.df[term] += 1
        self.tf = [Counter(t) for t in self.tok]

    def search_top1(self, query: str) -> str | None:
        q_terms = tokenize(query)
        best, best_s = None, 0.0
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
            if s > best_s:
                best_s, best = s, self.docs[i][0]
        return best


def rrf_fuse(rankings: list[list[str]], k: int = 60) -> list[str]:
    score = defaultdict(float)
    for r in rankings:
        for rank, p in enumerate(r):
            score[p] += 1.0 / (k + rank + 1)
    return [p for p, _ in sorted(score.items(), key=lambda x: -x[1])]


# ----- Per-cell runner -------------------------------------------------------

def run_cell(model: str, n_sentences: int, queries: list[tuple[int, str]],
             EXPECTED: dict, hit_fn, files: list[tuple[str, Path]]) -> dict:
    cell_id = f"{model.replace(':', '-')}_n{n_sentences}"
    print(f"\n{'='*70}\nCELL {cell_id} (model={model}, n_sent={n_sentences})\n{'='*70}", flush=True)

    cache_path = OUT_DIR / f"cache_{cell_id}.json"
    cache: dict[str, str] = {}
    if cache_path.exists() and os.environ.get("ABLATION_RESUME"):
        cache = json.loads(cache_path.read_text())
        print(f"[cache] resuming with {len(cache)} existing entries", flush=True)

    # ---- 1. Summarise ----
    system = build_system_prompt(n_sentences)
    t0 = time.time()
    n_skipped = 0
    for i, (rel, p) in enumerate(files):
        if rel in cache:
            n_skipped += 1
            continue
        content = read_text(p)
        if not content:
            n_skipped += 1
            continue
        try:
            sm = ollama_chat(model, system, build_user_msg(rel, content),
                             num_predict=max(150, n_sentences * 80))
        except Exception as e:
            print(f"  WARN: {rel} -> {e}", flush=True)
            sm = ""
        if not sm.strip():
            n_skipped += 1
            continue
        cache[rel] = sm
        if (i + 1) % 50 == 0 or i + 1 == len(files):
            cache_path.write_text(json.dumps(cache, indent=1))
            print(f"  [{i+1}/{len(files)}] cached={len(cache)} elapsed={time.time()-t0:.0f}s", flush=True)
    cache_path.write_text(json.dumps(cache, indent=1))
    summ_total_s = time.time() - t0
    summ_n = len(cache)
    summ_bytes = sum(len(v.encode("utf-8")) for v in cache.values())
    summ_words = sum(len(v.split()) for v in cache.values())
    print(f"[summary] {summ_n} files in {summ_total_s:.0f}s = {summ_total_s/max(summ_n,1):.1f}s/file", flush=True)
    print(f"[summary] total {summ_bytes} bytes ({summ_bytes//max(summ_n,1)} avg), "
          f"{summ_words} words ({summ_words//max(summ_n,1)} avg)", flush=True)

    # Build (path, summary) docs in stable order
    docs = sorted(cache.items())   # [(path, summary), ...]

    # ---- 2a. BM25 index ----
    print(f"[bm25] building index over {len(docs)} docs", flush=True)
    bm25 = BM25(docs)

    # ---- 2b. Qdrant index (in-process) ----
    print(f"[qdrant] embedding + indexing", flush=True)
    warmup_embedder()
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qm
    qpath = OUT_DIR / f"qdrant_{cell_id}"
    if qpath.exists():
        import shutil; shutil.rmtree(qpath)
    qc = QdrantClient(path=str(qpath))
    embs: dict[str, list[float]] = {}
    t1 = time.time()
    for i, (p, s) in enumerate(docs):
        e = ollama_embed(f"{p}\n{s}")
        if e is None:
            continue
        embs[p] = e
        if (i + 1) % 100 == 0:
            print(f"  embed [{i+1}/{len(docs)}] elapsed={time.time()-t1:.0f}s", flush=True)
    embed_s = time.time() - t1
    if embs:
        dim = len(next(iter(embs.values())))
        qc.create_collection(
            collection_name="cache",
            vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
        )
        points = [
            qm.PointStruct(id=i, vector=v, payload={"path": p})
            for i, (p, v) in enumerate(embs.items())
        ]
        qc.upsert(collection_name="cache", points=points)
    print(f"[qdrant] {len(embs)} embeddings in {embed_s:.0f}s", flush=True)

    # ---- 2c. ES index ----
    print(f"[es] indexing into elasticsearch", flush=True)
    from elasticsearch import Elasticsearch
    es = Elasticsearch(ES_URL, request_timeout=120, verify_certs=False,
                       basic_auth=(os.environ.get("ES_USER", "elastic"),
                                   os.environ.get("ES_PASS", "changeme"))
                       if os.environ.get("ES_USER") else None)
    es_index = f"abl_{cell_id.replace('.', '_').replace('-', '_')}"
    if es.indices.exists(index=es_index):
        es.indices.delete(index=es_index)
    es.indices.create(
        index=es_index,
        body={
            "mappings": {
                "properties": {
                    "path": {"type": "keyword"},
                    "text": {"type": "text"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": len(next(iter(embs.values()))) if embs else 768,
                        "index": True,
                        "similarity": "cosine",
                    },
                }
            }
        },
    )
    # bulk index
    bulk = []
    for p, s in docs:
        bulk.append({"index": {"_index": es_index}})
        doc = {"path": p, "text": f"{p}\n{s}"}
        if p in embs:
            doc["embedding"] = embs[p]
        bulk.append(doc)
    if bulk:
        es.bulk(operations=bulk, refresh=True)
    print(f"[es] indexed {len(docs)} docs into {es_index}", flush=True)

    # ---- 3. Score 18 queries on each backend ----
    backend_results = {
        "bm25":              {"hits": 0, "n": 0, "per_query": [], "total_s": 0.0},
        "qdrant":            {"hits": 0, "n": 0, "per_query": [], "total_s": 0.0},
        "elasticsearch-rrf": {"hits": 0, "n": 0, "per_query": [], "total_s": 0.0},
    }

    for qid, q in queries:
        if qid not in EXPECTED:
            continue
        exp = EXPECTED[qid]

        # BM25
        t = time.time()
        bm_top = bm25.search_top1(q)
        backend_results["bm25"]["total_s"] += time.time() - t
        bm_hit = bool(bm_top and hit_fn(bm_top, exp))
        backend_results["bm25"]["hits"] += int(bm_hit)
        backend_results["bm25"]["n"] += 1
        backend_results["bm25"]["per_query"].append(
            {"id": qid, "answer_path": bm_top, "hit": bm_hit})

        # Qdrant
        t = time.time()
        qv = ollama_embed(q)
        qd_top = None
        if qv is not None and embs:
            hits = qc.query_points(collection_name="cache", query=qv, limit=1).points
            qd_top = hits[0].payload["path"] if hits else None
        backend_results["qdrant"]["total_s"] += time.time() - t
        qd_hit = bool(qd_top and hit_fn(qd_top, exp))
        backend_results["qdrant"]["hits"] += int(qd_hit)
        backend_results["qdrant"]["n"] += 1
        backend_results["qdrant"]["per_query"].append(
            {"id": qid, "answer_path": qd_top, "hit": qd_hit})

        # ES-RRF: lexical + knn fused
        t = time.time()
        kw_hits = es.search(
            index=es_index,
            query={"match": {"text": q}},
            size=10, _source=["path"],
        )
        kw_rank = [h["_source"]["path"] for h in kw_hits["hits"]["hits"]]
        vec_rank = []
        if qv is not None:
            vec_hits = es.search(
                index=es_index,
                knn={"field": "embedding", "query_vector": qv,
                     "k": 10, "num_candidates": 50},
                size=10, _source=["path"],
            )
            vec_rank = [h["_source"]["path"] for h in vec_hits["hits"]["hits"]]
        rrf = rrf_fuse([kw_rank, vec_rank])
        es_top = rrf[0] if rrf else None
        backend_results["elasticsearch-rrf"]["total_s"] += time.time() - t
        es_hit = bool(es_top and hit_fn(es_top, exp))
        backend_results["elasticsearch-rrf"]["hits"] += int(es_hit)
        backend_results["elasticsearch-rrf"]["n"] += 1
        backend_results["elasticsearch-rrf"]["per_query"].append(
            {"id": qid, "answer_path": es_top, "hit": es_hit})

    es.indices.delete(index=es_index)
    qc.close()

    for bk, r in backend_results.items():
        pct = 100 * r["hits"] / max(1, r["n"])
        print(f"  {bk}: {r['hits']}/{r['n']} ({pct:.0f}%)  avg {r['total_s']/max(1,r['n']):.2f}s/query", flush=True)

    return {
        "cell_id": cell_id,
        "model": model,
        "n_sentences": n_sentences,
        "n_files_summarized": summ_n,
        "summary_total_bytes": summ_bytes,
        "summary_mean_bytes": summ_bytes // max(summ_n, 1),
        "summary_total_words": summ_words,
        "summary_mean_words": summ_words // max(summ_n, 1),
        "summarization_total_s": summ_total_s,
        "summarization_per_file_s": summ_total_s / max(summ_n, 1),
        "embedding_total_s": embed_s,
        "per_backend": backend_results,
    }


# ----- Main ------------------------------------------------------------------

CELLS_DEFAULT = [
    # Model sweep at fixed n_sentences=10 (per supervisor request).
    ("llama3.2:3b",  10),
    ("qwen2.5:7b",   10),
    ("llama3.1:8b",  10),
    ("qwen2.5:14b",  10),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="Single-cell mode: model name")
    ap.add_argument("--n-sentences", type=int, help="Single-cell mode: sentence count")
    ap.add_argument("--out", default=str(HERE / "fresh_results_ablation.json"))
    args = ap.parse_args()

    queries = parse_queries()
    EXPECTED, hit_fn = load_grader()
    queries = [(qid, q) for qid, q in queries if qid in EXPECTED]
    print(f"using {len(queries)} graded queries", flush=True)

    files = list(iter_files(REPO_ROOT))
    print(f"corpus: {len(files)} files", flush=True)

    if args.model and args.n_sentences:
        cells = [(args.model, args.n_sentences)]
    else:
        cells = CELLS_DEFAULT

    results = []
    for model, n in cells:
        try:
            r = run_cell(model, n, queries, EXPECTED, hit_fn, files)
        except Exception as e:
            import traceback; traceback.print_exc()
            r = {"cell_id": f"{model}_n{n}", "error": str(e)}
        results.append(r)
        # Snapshot results after each cell.
        Path(args.out).write_text(json.dumps({"cells": results}, indent=2))
        print(f"\n=> wrote {args.out} ({len(results)}/{len(cells)} cells)", flush=True)

    print("\n=== DONE ===", flush=True)
    for r in results:
        if "error" in r:
            print(f"  {r['cell_id']}: ERROR {r['error']}")
            continue
        per_bk = " | ".join(
            f"{bk} {b['hits']}/{b['n']}"
            for bk, b in r["per_backend"].items()
        )
        print(f"  {r['cell_id']}  bytes={r['summary_mean_bytes']:>4}  "
              f"sumtime={r['summarization_total_s']:.0f}s  {per_bk}")


if __name__ == "__main__":
    main()
