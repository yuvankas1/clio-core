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

#ifndef CLIO_CAE_CORE_BASE_OPERATOR_H_
#define CLIO_CAE_CORE_BASE_OPERATOR_H_

#include <chrono>
#include <ctime>
#include <string>

#include <clio_cte/core/core_client.h>

namespace clio::cae::core {

/**
 * BaseOperator - Abstract interface for data operators
 *
 * Operators transform data already stored in CTE. Unlike assimilators
 * (which ingest external data into CTE), operators read existing blobs,
 * process them, and write new blobs back to CTE.
 *
 * Idempotency contract (Path B-Full):
 *   - Operators that opt in to caching override Version(), OutputBlobName(),
 *     and ComputeInputHash().
 *   - The OperatorScheduler calls IsCached(tag) before Execute(); if it
 *     returns true, Execute() is skipped. Operators that do NOT override
 *     the new virtuals get the safe defaults (empty output name → IsCached
 *     returns false → Execute always runs), preserving legacy behavior.
 */
class BaseOperator {
 public:
  virtual ~BaseOperator() = default;

  /**
   * Execute the operator on a CTE tag
   * @param tag_name Name of the CTE tag containing input blobs
   * @return 0 on success, negative error code on failure
   */
  virtual int Execute(const std::string& tag_name) = 0;

  // ---------------------------------------------------------------------
  // Idempotency contract — operators override these to opt in to caching
  // ---------------------------------------------------------------------

  /**
   * Operator-implementation version. Bump on logic changes (e.g. summary
   * format change) to invalidate all previously-cached outputs. Default
   * 0 — concrete operators should override.
   */
  virtual int Version() const { return 0; }

  /**
   * Name of the blob this operator writes (e.g. "summary"). When empty,
   * the operator opts out of idempotency (Execute always runs).
   */
  virtual std::string OutputBlobName() const { return ""; }

  /**
   * Compute a content hash of this operator's inputs for the given tag.
   * The hash must include EVERY piece of state that affects the output:
   * input blob bytes, prompt text, model id, hyperparameters, etc. Two
   * Execute() calls producing identical output must produce identical
   * hashes.
   *
   * Default returns empty string → IsCached() returns false → Execute()
   * always runs (safe default for non-idempotent operators).
   */
  virtual std::string ComputeInputHash(clio::cte::core::Tag& tag) const {
    (void)tag;
    return "";
  }

  /**
   * Returns true if Execute() can be safely skipped because the output
   * blob already exists with metadata matching the current input hash and
   * operator version.
   *
   * Algorithm:
   *   1. Get OutputBlobName(); if empty → not cacheable, return false.
   *   2. Get ComputeInputHash(); if empty → not cacheable, return false.
   *   3. Read tag.GetBlobMeta(output). If op_version doesn't match
   *      Version() → stale, return false.
   *   4. If meta.input_hash equals freshly-computed hash → cached, return
   *      true. Else → return false.
   */
  bool IsCached(clio::cte::core::Tag& tag) const {
    std::string output = OutputBlobName();
    if (output.empty()) return false;
    std::string fresh_hash = ComputeInputHash(tag);
    if (fresh_hash.empty()) return false;
    if (!tag.HasBlobMeta(output)) return false;
    auto meta = tag.GetBlobMeta(output);
    if (meta.op_version != Version()) return false;
    return meta.input_hash == fresh_hash;
  }

  /**
   * Current UTC time as an ISO 8601 string ("2026-05-04T14:32:18Z").
   * Convenience for setting BlobMeta::created_at when an operator writes
   * a new output blob.
   */
  static std::string CurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm_utc{};
#if defined(_WIN32)
    gmtime_s(&tm_utc, &t);
#else
    gmtime_r(&t, &tm_utc);
#endif
    char buf[25];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm_utc);
    return std::string(buf);
  }
};

}  // namespace clio::cae::core

#endif  // CLIO_CAE_CORE_BASE_OPERATOR_H_
