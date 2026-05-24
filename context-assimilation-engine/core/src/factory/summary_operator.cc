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

#include <clio_runtime/clio_runtime.h>
#include <clio_cae/core/factory/summary_operator.h>

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <clio_cae/core/factory/hashing.h>

#ifdef CLIO_CAE_ENABLE_SUMMARY_OP
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#endif

// Include wrp_cte headers after closing any wrp_cae namespace to avoid Method
// namespace collision
#include <clio_cte/core/core_client.h>

namespace clio::cae::core {

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

SummaryOperator::Config SummaryOperator::Config::FromEnv() {
  Config c;
  if (const char* e = std::getenv("CAE_SUMMARY_ENDPOINT")) c.endpoint = e;
  if (const char* m = std::getenv("CAE_SUMMARY_MODEL")) c.model = m;
  // system_prompt and max_tokens are resolved lazily by ResolveSystemPrompt/
  // ResolveMaxTokens so the legacy env-on-every-call semantics are preserved.
  return c;
}

// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------

SummaryOperator::SummaryOperator(
    std::shared_ptr<clio::cte::core::Client> cte_client, Config config)
    : cte_client_(std::move(cte_client)), config_(std::move(config)) {}

SummaryOperator::SummaryOperator(
    std::shared_ptr<clio::cte::core::Client> cte_client)
    : SummaryOperator(std::move(cte_client), Config::FromEnv()) {}

// ---------------------------------------------------------------------------
// Prompt + token resolution
// ---------------------------------------------------------------------------

bool SummaryOperator::HasHumanDescription(const std::string& description) {
  return description.find("description:") != std::string::npos ||
         description.find("description=") != std::string::npos ||
         description.find("long_name:") != std::string::npos;
}

std::string SummaryOperator::ResolveSystemPrompt(
    const std::string& description) const {
  if (!config_.system_prompt.empty()) return config_.system_prompt;
  if (const char* p = std::getenv("CAE_SUMMARY_SYSTEM_PROMPT")) return p;
  if (HasHumanDescription(description)) {
    return "You are a scientific data analyst. Given a dataset description "
           "from a simulation output file, summarize it in exactly 4 to 8 "
           "words. Keep domain-specific terms. Return ONLY the summary, "
           "nothing else.";
  }
  return "You are a scientific data analyst for HPC simulations. Given raw "
         "metadata from a simulation output file, write a concise 4-8 word "
         "searchable description. Identify the simulation type and key "
         "properties. Translate numeric codes and flags to their scientific "
         "meaning. Return ONLY the description, nothing else.";
}

int SummaryOperator::ResolveMaxTokens() const {
  if (config_.max_tokens > 0) return config_.max_tokens;
  if (const char* mt = std::getenv("CAE_SUMMARY_MAX_TOKENS")) {
    int v = std::atoi(mt);
    if (v > 0 && v <= 4096) return v;
  }
  return 64;
}

// ---------------------------------------------------------------------------
// Idempotency input hash
// ---------------------------------------------------------------------------

std::string SummaryOperator::ComputeInputHash(clio::cte::core::Tag& tag) const {
  chi::u64 sz = tag.GetBlobSize("description");
  if (sz == 0) return "";
  std::vector<char> buf(sz);
  tag.GetBlob("description", buf.data(), sz);
  std::string description(buf.data(), sz);

  std::string prompt = ResolveSystemPrompt(description);
  int max_tokens = ResolveMaxTokens();

  // Hash: description bytes, resolved prompt, model, max_tokens, op_version.
  // Any change to any of these produces a different hash → cache miss → re-run.
  std::string h = hashing::Fnv1a64Hex(description);
  h = hashing::ChainHex(h, prompt);
  h = hashing::ChainHex(h, config_.model);
  h = hashing::ChainHex(h, std::to_string(max_tokens));
  h = hashing::ChainHex(h, std::to_string(Version()));
  return h;
}

// ---------------------------------------------------------------------------
// Execute
// ---------------------------------------------------------------------------

int SummaryOperator::Execute(const std::string& tag_name) {
  HLOG(kInfo, "SummaryOperator::Execute ENTRY: tag='{}'", tag_name);

  // Validate configuration
  if (config_.endpoint.empty() || config_.model.empty()) {
    HLOG(kError,
         "SummaryOperator: endpoint and model must be set (Config or "
         "CAE_SUMMARY_ENDPOINT/CAE_SUMMARY_MODEL env vars)");
    return -1;
  }

  // Idempotency check: skip LLM call if a prior run produced the same output
  // for this exact (description, prompt, model, max_tokens, op_version).
  try {
    clio::cte::core::Tag tag(tag_name);
    if (IsCached(tag)) {
      HLOG(kInfo, "SummaryOperator: cache HIT for tag '{}' — skipping LLM call",
           tag_name);
      return 0;
    }
  } catch (const std::exception& e) {
    HLOG(kError, "SummaryOperator: Failed to open tag '{}' for cache check: {}",
         tag_name, e.what());
    return -2;
  }

  // Step 1: Read the description blob
  std::string description = ReadDescriptionBlob(tag_name);
  if (description.empty()) {
    HLOG(kError, "SummaryOperator: Failed to read description blob from tag '{}'",
         tag_name);
    return -3;
  }
  HLOG(kInfo, "SummaryOperator: Read description: '{}'", description);

  // Step 2: Call LLM (prompt is resolved per call via ResolveSystemPrompt)
  std::string summary = CallLlm(description);
  if (summary.empty()) {
    HLOG(kError, "SummaryOperator: LLM call failed for tag '{}'", tag_name);
    return -4;
  }
  HLOG(kInfo, "SummaryOperator: Generated summary: '{}'", summary);

  // Step 3: Write summary blob
  int rc = WriteSummaryBlob(tag_name, summary);
  if (rc != 0) {
    HLOG(kError, "SummaryOperator: Failed to write summary blob to tag '{}'",
         tag_name);
    return rc;
  }

  // Step 4: Write idempotency + provenance metadata.
  // Category is propagated from the description blob's meta when present, so
  // queries can filter results by data category (code / scientific / ...).
  try {
    clio::cte::core::Tag tag(tag_name);
    clio::cte::core::BlobMeta meta;
    meta.input_hash = ComputeInputHash(tag);
    meta.op_version = Version();
    meta.prompt_hash = hashing::Fnv1a64Hex(ResolveSystemPrompt(description));
    meta.model_id = config_.model;
    meta.created_at = CurrentTimestamp();
    auto desc_meta = tag.GetBlobMeta("description");
    meta.category = desc_meta.category;
    tag.PutBlobMeta("summary", meta);
  } catch (const std::exception& e) {
    // Meta write failure is non-fatal: the summary blob is already saved.
    // Next call will see no meta → IsCached returns false → re-run. That's
    // wasteful but correct.
    HLOG(kError, "SummaryOperator: Failed to write summary meta for '{}': {} "
                 "(non-fatal; will trigger re-run on next call)",
         tag_name, e.what());
  }

  HLOG(kInfo, "SummaryOperator::Execute EXIT: Success for tag '{}'", tag_name);
  return 0;
}

std::string SummaryOperator::ReadDescriptionBlob(const std::string& tag_name) {
  try {
    clio::cte::core::Tag tag(tag_name);

    // Get the size of the description blob
    chi::u64 blob_size = tag.GetBlobSize("description");
    if (blob_size == 0) {
      HLOG(kError, "SummaryOperator: 'description' blob not found or empty in "
                    "tag '{}'",
           tag_name);
      return "";
    }

    // Read the blob data
    std::vector<char> buffer(blob_size);
    tag.GetBlob("description", buffer.data(), blob_size);

    return std::string(buffer.data(), blob_size);
  } catch (const std::exception& e) {
    HLOG(kError, "SummaryOperator: Exception reading description blob: {}",
         e.what());
    return "";
  }
}

#ifdef CLIO_CAE_ENABLE_SUMMARY_OP

// libcurl write callback
static size_t CurlWriteCallback(void* contents, size_t size, size_t nmemb,
                                 std::string* output) {
  size_t total_size = size * nmemb;
  output->append(static_cast<char*>(contents), total_size);
  return total_size;
}

std::string SummaryOperator::CallLlm(const std::string& description) const {
  // Resolve prompt + max_tokens via the unified resolution helpers so that
  // ComputeInputHash and CallLlm always agree on the actual values used.
  std::string system_prompt = ResolveSystemPrompt(description);
  int max_tokens = ResolveMaxTokens();

  nlohmann::json request_body;
  request_body["model"] = config_.model;
  request_body["messages"] = nlohmann::json::array({
      {{"role", "system"}, {"content", system_prompt}},
      {{"role", "user"}, {"content", description}},
  });
  request_body["max_tokens"] = max_tokens;
  request_body["temperature"] = 0.0;

  std::string payload = request_body.dump();
  std::string url = config_.endpoint + "/chat/completions";

  HLOG(kDebug, "SummaryOperator: POST {} payload={}", url, payload);

  // Initialize curl
  CURL* curl = curl_easy_init();
  if (!curl) {
    HLOG(kError, "SummaryOperator: Failed to initialize libcurl");
    return "";
  }

  std::string response_body;
  struct curl_slist* headers = nullptr;
  headers = curl_slist_append(headers, "Content-Type: application/json");
  headers = curl_slist_append(headers, "Accept: application/json");

  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload.c_str());
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, CurlWriteCallback);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_body);
  curl_easy_setopt(curl, CURLOPT_TIMEOUT, 60L);

  CURLcode res = curl_easy_perform(curl);

  curl_slist_free_all(headers);
  curl_easy_cleanup(curl);

  if (res != CURLE_OK) {
    HLOG(kError, "SummaryOperator: curl request failed: {}",
         curl_easy_strerror(res));
    return "";
  }

  HLOG(kDebug, "SummaryOperator: LLM response: {}", response_body);

  // Parse the response
  try {
    nlohmann::json response = nlohmann::json::parse(response_body);
    if (response.contains("choices") && !response["choices"].empty()) {
      auto& message = response["choices"][0]["message"];
      // Try "content" first, fall back to "reasoning_content" (Qwen3 thinking mode)
      std::string result;
      if (message.contains("content") && !message["content"].is_null()) {
        result = message["content"].get<std::string>();
      }
      if (result.empty() && message.contains("reasoning_content") &&
          !message["reasoning_content"].is_null()) {
        HLOG(kDebug, "SummaryOperator: Using reasoning_content (thinking mode)");
        result = message["reasoning_content"].get<std::string>();
      }
      if (!result.empty()) {
        return result;
      }
      HLOG(kError, "SummaryOperator: Both content and reasoning_content are empty");
      return "";
    }
    HLOG(kError, "SummaryOperator: No 'choices' in LLM response");
    return "";
  } catch (const std::exception& e) {
    HLOG(kError, "SummaryOperator: Failed to parse LLM response: {}",
         e.what());
    return "";
  }
}

#else  // !CLIO_CAE_ENABLE_SUMMARY_OP

std::string SummaryOperator::CallLlm(const std::string& description) const {
  (void)description;
  HLOG(kError,
       "SummaryOperator: Summary operator not compiled in. "
       "Rebuild with -DCLIO_CAE_ENABLE_SUMMARY_OP=ON");
  return "";
}

#endif  // CLIO_CAE_ENABLE_SUMMARY_OP

int SummaryOperator::WriteSummaryBlob(const std::string& tag_name,
                                      const std::string& summary) {
  try {
    clio::cte::core::Tag tag(tag_name);
    tag.PutBlob("summary", summary.c_str(), summary.size());
    HLOG(kDebug, "SummaryOperator: Wrote 'summary' blob ({} bytes) to tag '{}'",
         summary.size(), tag_name);
    return 0;
  } catch (const std::exception& e) {
    HLOG(kError, "SummaryOperator: Exception writing summary blob: {}",
         e.what());
    return -5;
  }
}

}  // namespace clio::cae::core
