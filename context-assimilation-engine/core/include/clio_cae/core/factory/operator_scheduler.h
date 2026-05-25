/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * OperatorScheduler — orchestrates the operator chain that runs after data
 * has been assimilated into a CTE tag. For B-Full it runs:
 *
 *   1. SummaryOperator  (idempotent: skips if summary blob meta matches
 *                        a freshly-computed input hash)
 *   2. UpdateKnowledgeGraph  (CTE chimaera task that resolves depth policy
 *                             via depth_controller_ and inserts/upserts
 *                             into the configured KGBackend)
 *
 * The chain is intentionally hardcoded for now. B6/B7 will lift it to a
 * configurable per-category operator chain driven by compose YAML.
 *
 * Idempotency: every step in the chain is independently idempotent, so the
 * scheduler can be called repeatedly on the same tag with near-zero cost
 * when nothing has changed. That's what makes periodic re-scan (B11) cheap.
 */

#ifndef CLIO_CAE_CORE_OPERATOR_SCHEDULER_H_
#define CLIO_CAE_CORE_OPERATOR_SCHEDULER_H_

#include <memory>
#include <string>

#include <clio_cae/core/factory/summary_operator.h>

namespace clio::cte::core {
class Client;
}  // namespace clio::cte::core

namespace clio::cae::core {

class OperatorScheduler {
 public:
  /**
   * @param cte_client    Client used to talk to CTE (for tag operations and
   *                      the UpdateKnowledgeGraph chimaera task).
   * @param summary_cfg   Configuration for the SummaryOperator step. If
   *                      empty / unset fields, falls back to env vars (legacy
   *                      behavior). Pass a Config with system_prompt set to
   *                      drive code-tuned vs scientific-tuned summarization.
   */
  OperatorScheduler(std::shared_ptr<clio::cte::core::Client> cte_client,
                    SummaryOperator::Config summary_cfg = {});

  /**
   * Run the operator chain for a tag that has just been assimilated.
   * Each step is idempotent — calling RunForTag on a tag whose source has
   * not changed since the last call is cheap (hash compares + UPSERT).
   *
   * @param tag_name CTE tag name (typically the file path).
   * @return 0 on success; negative on hard failure. Soft skips (e.g.
   *         SummaryOperator not configured) are not errors and proceed
   *         with empty summary so the KG can still index path metadata.
   */
  int RunForTag(const std::string& tag_name);

  // Result codes:
  //   0       success
  //   -10     GetOrCreateTag failed
  //   -11     UpdateKnowledgeGraph chimaera task failed
  //   <other> propagated from SummaryOperator::Execute (-1 endpoint/model
  //           unset is treated as a soft skip, NOT propagated)

 private:
  std::shared_ptr<clio::cte::core::Client> cte_client_;
  SummaryOperator::Config summary_cfg_;
};

}  // namespace clio::cae::core

#endif  // CLIO_CAE_CORE_OPERATOR_SCHEDULER_H_
