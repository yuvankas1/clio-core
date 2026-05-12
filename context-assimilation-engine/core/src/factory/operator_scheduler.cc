/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * OperatorScheduler implementation. See header for design notes.
 */

#include <wrp_cae/core/factory/operator_scheduler.h>

#include <vector>

#include <hermes_shm/util/logging.h>
#include <wrp_cte/core/core_client.h>

namespace wrp_cae::core {

OperatorScheduler::OperatorScheduler(
    std::shared_ptr<wrp_cte::core::Client> cte_client,
    SummaryOperator::Config summary_cfg)
    : cte_client_(std::move(cte_client)),
      summary_cfg_(std::move(summary_cfg)) {}

int OperatorScheduler::RunForTag(const std::string& tag_name) {
  HLOG(kInfo, "OperatorScheduler::RunForTag ENTRY: tag='{}'", tag_name);

  // Step 1 — SummaryOperator. Idempotent: skips LLM call if cached.
  // Soft-skip behavior: if endpoint/model are unset (rc == -1), proceed
  // with empty summary so the KG still indexes path metadata (L0/L1 mode).
  SummaryOperator summary_op(cte_client_, summary_cfg_);
  int summary_rc = summary_op.Execute(tag_name);
  if (summary_rc != 0 && summary_rc != -1) {
    HLOG(kError,
         "OperatorScheduler: SummaryOperator failed for tag '{}' rc={}",
         tag_name, summary_rc);
    return summary_rc;
  }
  if (summary_rc == -1) {
    HLOG(kInfo,
         "OperatorScheduler: SummaryOperator not configured — proceeding "
         "with empty summary (L0/L1 mode) for tag '{}'", tag_name);
  }

  // Step 2 — Read the produced summary blob (if any) so we can pass it to
  // CTE's UpdateKnowledgeGraph task. If no summary was produced (soft
  // skip), pass empty string — CTE will index only path metadata.
  std::string summary;
  wrp_cte::core::TagId tag_id;
  try {
    wrp_cte::core::Tag tag(tag_name);
    tag_id = tag.GetTagId();
    chi::u64 sz = tag.GetBlobSize("summary");
    if (sz > 0 && sz < 64 * 1024) {
      std::vector<char> buf(static_cast<size_t>(sz));
      tag.GetBlob("summary", buf.data(), sz);
      summary.assign(buf.data(), buf.size());
    }
  } catch (const std::exception& e) {
    HLOG(kError,
         "OperatorScheduler: Failed to read summary blob for tag '{}': {}",
         tag_name, e.what());
    return -10;
  }

  // Step 3 — Trigger CTE's existing UpdateKnowledgeGraph chimaera task.
  // CTE resolves depth policy via depth_controller_ and upserts into the
  // configured KGBackend (bm25 / elasticsearch / qdrant / neo4j).
  auto upd = cte_client_->AsyncUpdateKnowledgeGraph(tag_id, tag_name, summary);
  upd.Wait();
  if (upd->GetReturnCode() != 0) {
    HLOG(kError,
         "OperatorScheduler: UpdateKnowledgeGraph failed for tag '{}' rc={}",
         tag_name, upd->GetReturnCode());
    return -11;
  }

  HLOG(kInfo, "OperatorScheduler::RunForTag EXIT: tag='{}' (summary {} bytes)",
       tag_name, summary.size());
  return 0;
}

}  // namespace wrp_cae::core
