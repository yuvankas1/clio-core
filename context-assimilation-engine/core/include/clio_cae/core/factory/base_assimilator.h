/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef CLIO_CAE_CORE_BASE_ASSIMILATOR_H_
#define CLIO_CAE_CORE_BASE_ASSIMILATOR_H_

#include <algorithm>
#include <cctype>
#include <string>

#include <clio_cae/core/factory/assimilation_ctx.h>
#include <clio_cte/core/core_client.h>
#include <clio_runtime/task.h>

namespace clio::cae::core {

/**
 * CategoryFromPath - Map a file path to a coarse data category, based on
 * the file extension. Used by assimilators to label the description blob
 * with BlobMeta.category. Today only the label is set — no per-category
 * routing yet. The label exists so deployments can later inspect category
 * tags, and so future code (B6/B8 if ever re-introduced) can route on it.
 *
 *   "code"       — source code (.cc .cpp .h .py .go .rs .js .ts .md ...)
 *   "scientific" — HPC datasets (.h5 .nc .hdf .hdf5)
 *   "document"   — plain text / PDF (.txt .pdf)
 *   "other"      — fallback for anything not recognized
 */
inline std::string CategoryFromPath(const std::string& path) {
  // Find the rightmost '.' after the last '/' or '\\'
  size_t slash = path.find_last_of("/\\");
  size_t dot = path.find_last_of('.');
  if (dot == std::string::npos ||
      (slash != std::string::npos && dot < slash)) {
    return "other";
  }
  std::string ext = path.substr(dot + 1);
  std::transform(ext.begin(), ext.end(), ext.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  // Code: common source-code, build-system, and project-doc extensions.
  static const std::string code_exts[] = {
      "c", "cc", "cpp", "cxx", "h", "hh", "hpp", "hxx",
      "py", "go", "rs", "js", "ts", "tsx", "jsx",
      "java", "kt", "swift", "rb", "php", "scala",
      "md", "rst", "txt", "sh", "bash", "zsh",
      "yaml", "yml", "toml", "json", "xml",
      "cmake", "make", "mk", "dockerfile"};
  for (const auto& e : code_exts) {
    if (ext == e) return "code";
  }

  // HPC scientific data formats.
  if (ext == "h5" || ext == "hdf" || ext == "hdf5" || ext == "nc" ||
      ext == "netcdf") {
    return "scientific";
  }

  // Documents (PDF only — most "text" files above are already code-tagged).
  if (ext == "pdf") return "document";

  return "other";
}

/**
 * BaseAssimilator - Abstract interface for data assimilators
 * Concrete implementations handle different data sources (file, URL, etc.)
 *
 * NOTE: Schedule is a coroutine that must be co_awaited from runtime code.
 * The error code is returned via output parameter since coroutines return TaskResume.
 *
 * Idempotency contract (Path B-Full):
 *   - Assimilators that opt in to caching override Version() and
 *     ComputeContentHash().
 *   - Before reading the source bytes, an assimilator should call
 *     IsCached(tag, fresh_content_hash). If true, Schedule can short-
 *     circuit: the description blob is already current, no I/O needed.
 *   - Assimilators that do NOT override the new virtuals (default
 *     content_hash = "") get IsCached() == false and always re-ingest,
 *     preserving legacy behavior.
 */
class BaseAssimilator {
 public:
  virtual ~BaseAssimilator() = default;

  /**
   * Schedule assimilation tasks based on the provided context
   * This is a coroutine that uses co_await for async CTE operations.
   * @param ctx Assimilation context with source, destination, and metadata
   * @param error_code Output: 0 on success, non-zero error code on failure
   * @return TaskResume for coroutine suspension/resumption
   */
  virtual chi::TaskResume Schedule(const AssimilationCtx& ctx, int& error_code) = 0;

  // ---------------------------------------------------------------------
  // Idempotency contract — assimilators override these to opt in to caching
  // ---------------------------------------------------------------------

  /**
   * Assimilator-implementation version. Bump on logic changes (e.g. new
   * format extraction, different chunking) to invalidate cached
   * description blobs. Default 0 — concrete assimilators should override.
   */
  virtual int Version() const { return 0; }

  /**
   * Name of the blob this assimilator writes as the canonical input to
   * downstream operators. Defaults to "description" — most assimilators
   * write the source bytes / extracted text to a blob of that name.
   */
  virtual std::string OutputBlobName() const { return "description"; }

  /**
   * Compute a content hash of the source bytes referenced by ctx. For
   * file-based assimilators this is typically a hash of the file
   * contents. For URL-based assimilators it might combine
   * Content-Length + Last-Modified or an ETag.
   *
   * Default returns empty string → IsCached() returns false → Schedule()
   * always re-ingests (safe default).
   */
  virtual std::string ComputeContentHash(const AssimilationCtx& ctx) const {
    (void)ctx;
    return "";
  }

  /**
   * Returns true if Schedule() can be safely skipped because the
   * destination tag already has a description blob whose meta records a
   * matching content_hash and assimilator version.
   *
   * Assimilator subclasses call this at the top of Schedule(). When it
   * returns true, set error_code = 0 and co_return without doing any
   * I/O.
   */
  bool IsCached(clio::cte::core::Tag& tag,
                const std::string& content_hash) const {
    if (content_hash.empty()) return false;
    std::string output = OutputBlobName();
    if (output.empty()) return false;
    if (!tag.HasBlobMeta(output)) return false;
    auto meta = tag.GetBlobMeta(output);
    if (meta.op_version != Version()) return false;
    return meta.content_hash == content_hash;
  }
};

}  // namespace clio::cae::core

#endif  // CLIO_CAE_CORE_BASE_ASSIMILATOR_H_
