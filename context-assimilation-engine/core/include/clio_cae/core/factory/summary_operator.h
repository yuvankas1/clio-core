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

#ifndef CLIO_CAE_CORE_SUMMARY_OPERATOR_H_
#define CLIO_CAE_CORE_SUMMARY_OPERATOR_H_

#include <clio_cae/core/factory/base_operator.h>
#include <memory>
#include <string>

// Forward declaration
namespace clio::cte::core {
class Client;
}  // namespace clio::cte::core

namespace clio::cae::core {

/**
 * SummaryOperator - Summarizes a "description" blob using a local LLM
 *
 * Reads the "description" blob from a CTE tag, sends it to an
 * OpenAI-compatible inference endpoint, and writes a "summary" blob
 * back to the same tag along with idempotency metadata.
 *
 * Configuration: prefer the explicit Config-arg constructor. The legacy
 * single-arg constructor (kept for backward compatibility with
 * benchmarks) reads CAE_SUMMARY_* environment variables.
 *
 * Idempotency: this operator opts in to BaseOperator's IsCached()
 * protocol. The OperatorScheduler will skip Execute() when the
 * "summary" blob's stored meta.input_hash matches a freshly-computed
 * hash of (description bytes + prompt + model + max_tokens + op_version).
 *
 * Error codes:
 *   -1: Missing endpoint or model
 *   -2: Tag not found or could not be created
 *   -3: "description" blob not found in tag
 *   -4: LLM inference call failed
 *   -5: Failed to write "summary" blob
 */
class SummaryOperator : public BaseOperator {
 public:
  /** Explicit configuration. Any empty/zero field falls back to the
   *  corresponding CAE_SUMMARY_* env var, else a hardcoded default. */
  struct Config {
    std::string endpoint;       // OpenAI-compatible base URL
    std::string model;          // model id (e.g. "qwen2.5:7b")
    std::string system_prompt;  // when non-empty, used verbatim (no branching)
    int max_tokens = 0;         // 0 = use env or default

    /** Build a Config from CAE_SUMMARY_* environment variables. */
    static Config FromEnv();
  };

  /** Bump on logic changes that affect output to invalidate cached summaries. */
  static constexpr int kVersion = 1;

  /** Primary constructor: explicit config. */
  SummaryOperator(std::shared_ptr<clio::cte::core::Client> cte_client,
                  Config config);

  /** Legacy convenience constructor: equivalent to passing
   *  Config::FromEnv(). Existing benchmark/test callers continue to work
   *  unchanged. */
  explicit SummaryOperator(std::shared_ptr<clio::cte::core::Client> cte_client);

  int Execute(const std::string& tag_name) override;

  // BaseOperator idempotency contract:
  int Version() const override { return kVersion; }
  std::string OutputBlobName() const override { return "summary"; }
  std::string ComputeInputHash(clio::cte::core::Tag& tag) const override;

 private:
  std::string ReadDescriptionBlob(const std::string& tag_name);

  /** Decide between summarization vs interpretation prompt for HPC data
   *  in the legacy default path (when config_.system_prompt is empty). */
  static bool HasHumanDescription(const std::string& description);

  /** Resolve the actual system prompt to send to the LLM:
   *  config_.system_prompt > CAE_SUMMARY_SYSTEM_PROMPT env > HPC defaults. */
  std::string ResolveSystemPrompt(const std::string& description) const;

  /** Resolve max_tokens: config_.max_tokens > CAE_SUMMARY_MAX_TOKENS env > 64. */
  int ResolveMaxTokens() const;

  std::string CallLlm(const std::string& description) const;

  int WriteSummaryBlob(const std::string& tag_name, const std::string& summary);

  std::shared_ptr<clio::cte::core::Client> cte_client_;
  Config config_;
};

}  // namespace clio::cae::core

#endif  // CLIO_CAE_CORE_SUMMARY_OPERATOR_H_
