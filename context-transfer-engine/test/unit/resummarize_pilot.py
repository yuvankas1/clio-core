#!/usr/bin/env python3
"""Pilot: re-summarize the ~15 target files for our 20 queries with the v2
search-optimized prompt and patch the cache JSON in place.

Usage:
  python resummarize_pilot.py [--model qwen2.5:7b] [--dry-run]

Reads the v2 prompt from prompts/search_optimized_summary_v2.txt, walks
the target files, calls Ollama once per file with the new prompt, and
overwrites those entries in the most-recent acropolis_summary_cache_*.json.
The original cache is backed up to *.bak before patching.
"""

import argparse, glob, json, os, shutil, sys, time
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.environ.get("ACROPOLIS_CACHE_DIR", "C:/temp")
REPO_LOCAL = os.environ.get(
    "ACROPOLIS_REPO_LOCAL",
    "c:/Users/rajni/Documents/GPU_OS/clio-core")
OLLAMA_URL = os.environ.get("ACROPOLIS_OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = "qwen2.5:7b"

# (cache_key_path, local_disk_path) — cache uses /workspace/ paths.
TARGETS = [
    ("/workspace/context-transfer-engine/core/include/wrp_cte/core/kg_backend_qdrant.h",
     "context-transfer-engine/core/include/wrp_cte/core/kg_backend_qdrant.h"),
    ("/workspace/context-transfer-engine/core/include/wrp_cte/core/kg_backend_elasticsearch.h",
     "context-transfer-engine/core/include/wrp_cte/core/kg_backend_elasticsearch.h"),
    ("/workspace/context-transfer-engine/core/include/wrp_cte/core/kg_backend_neo4j.h",
     "context-transfer-engine/core/include/wrp_cte/core/kg_backend_neo4j.h"),
    ("/workspace/context-transfer-engine/core/include/wrp_cte/core/hdf5_summary.h",
     "context-transfer-engine/core/include/wrp_cte/core/hdf5_summary.h"),
    ("/workspace/tools/acropolis_depth/set_depth.cc",
     "tools/acropolis_depth/set_depth.cc"),
    ("/workspace/context-runtime/modules/MOD_NAME/test/test_gpu_submission_gpu.cc",
     "context-runtime/modules/MOD_NAME/test/test_gpu_submission_gpu.cc"),
    ("/workspace/iowarp-llm/weights/src/ggml_iowarp_backend.cc",
     "iowarp-llm/weights/src/ggml_iowarp_backend.cc"),
    ("/workspace/iowarp-llm/weights/include/wrp_llm/weights/ggml_iowarp_backend.h",
     "iowarp-llm/weights/include/wrp_llm/weights/ggml_iowarp_backend.h"),
    ("/workspace/iowarp-llm/kvcache/include/wrp_llm/kvcache/kvcache_manager.h",
     "iowarp-llm/kvcache/include/wrp_llm/kvcache/kvcache_manager.h"),
    ("/workspace/run_e2e_gpu_test.sh",
     "run_e2e_gpu_test.sh"),
    ("/workspace/context-transfer-engine/test/unit/bench_repo_scan.cc",
     "context-transfer-engine/test/unit/bench_repo_scan.cc"),
    ("/workspace/context-transfer-engine/core/include/wrp_cte/core/kg_backend_bm25.h",
     "context-transfer-engine/core/include/wrp_cte/core/kg_backend_bm25.h"),
    ("/workspace/context-transfer-engine/core/include/wrp_cte/core/depth_controller.h",
     "context-transfer-engine/core/include/wrp_cte/core/depth_controller.h"),
    ("/workspace/context-transfer-engine/core/include/wrp_cte/core/embedding_client.h",
     "context-transfer-engine/core/include/wrp_cte/core/embedding_client.h"),
    ("/workspace/context-transfer-engine/test/unit/test_indexing_depth_config.cc",
     "context-transfer-engine/test/unit/test_indexing_depth_config.cc"),
    ("/workspace/context-transfer-engine/core/include/wrp_cte/core/core_tasks.h",
     "context-transfer-engine/core/include/wrp_cte/core/core_tasks.h"),
    ("/workspace/context-assimilation-engine/core/src/factory/summary_operator.cc",
     "context-assimilation-engine/core/src/factory/summary_operator.cc"),
    ("/workspace/config/indexing_depth_defaults.yaml",
     "config/indexing_depth_defaults.yaml"),
]

MAX_FILE_CHARS = 8000   # truncate long files before sending to LLM
TIMEOUT_S = 120


def load_prompt():
    with open(os.path.join(HERE, "prompts", "search_optimized_summary_v2.txt"),
              encoding="utf-8") as f:
        return f.read().strip()


def find_cache():
    files = sorted(
        glob.glob(os.path.join(CACHE_DIR, "acropolis_summary_cache_*.json")),
        key=os.path.getmtime, reverse=True)
    return files[0] if files else None


def ollama_chat(model, system, user_text):
    body = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user_text},
        ],
        "options": {"temperature": 0.2, "num_predict": 300},
    }
    req = Request(
        f"{OLLAMA_URL}/api/chat",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST")
    try:
        with urlopen(req, timeout=TIMEOUT_S) as r:
            data = json.loads(r.read().decode("utf-8"))
        msg = data.get("message", {}).get("content", "")
        return msg.strip()
    except (URLError, HTTPError) as e:
        return f"[ERROR: {e}]"


def build_user_msg(local_path: str) -> str | None:
    full = os.path.join(REPO_LOCAL, local_path)
    if not os.path.exists(full):
        return None
    try:
        with open(full, encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception as e:
        return f"[unreadable: {e}]"
    if len(content) > MAX_FILE_CHARS:
        head = content[: MAX_FILE_CHARS // 2]
        tail = content[-MAX_FILE_CHARS // 2:]
        content = head + "\n\n... [truncated] ...\n\n" + tail
    return f"FILE PATH: {local_path}\n\nFILE CONTENT:\n```\n{content}\n```"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--dry-run", action="store_true",
                    help="Print summaries without patching the cache.")
    args = ap.parse_args()

    prompt = load_prompt()
    cache_path = find_cache()
    if not cache_path:
        print(f"ERROR: no cache found in {CACHE_DIR}", file=sys.stderr)
        sys.exit(1)
    print(f"Using cache: {cache_path}")

    with open(cache_path, encoding="utf-8") as f:
        cache = json.load(f)
    print(f"Cache entries: {len(cache)}")

    if not args.dry_run:
        backup = cache_path + ".bak"
        if not os.path.exists(backup):
            shutil.copy(cache_path, backup)
            print(f"Backup -> {backup}")

    new_summaries = {}
    for cache_key, local_path in TARGETS:
        print(f"\n--- {local_path} ---")
        user_msg = build_user_msg(local_path)
        if user_msg is None:
            print(f"  SKIP (file not on disk)")
            continue
        if user_msg.startswith("[unreadable"):
            print(f"  SKIP {user_msg}")
            continue
        t0 = time.time()
        summary = ollama_chat(args.model, prompt, user_msg)
        dt = time.time() - t0
        print(f"  generated in {dt:.1f}s ({len(summary)} chars):")
        print(f"  {summary[:400]}{'...' if len(summary) > 400 else ''}")

        old = cache.get(cache_key, "")
        if old:
            print(f"  (replacing {len(old)}-char old summary)")
        else:
            print(f"  (no prior entry — new key)")
        new_summaries[cache_key] = summary

    if args.dry_run:
        print(f"\n[dry-run] would patch {len(new_summaries)} entries")
        return

    cache.update(new_summaries)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)
    print(f"\nPatched {len(new_summaries)} entries in {cache_path}")
    print(f"Original backed up at {cache_path}.bak")


if __name__ == "__main__":
    main()
