/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * Portable, dependency-free hashing utilities used by CAE operators and
 * assimilators for content-addressed idempotency. FNV-1a 64-bit is chosen
 * because it is deterministic across stdlib/platform versions (unlike
 * std::hash), has good distribution for our cache-key use case, and needs
 * no external library.
 *
 * The hashes are NOT cryptographic. They are used as cache keys, not
 * security tokens — collision probability at our scale (<1M files) is
 * astronomically low.
 */

#ifndef CLIO_CAE_CORE_HASHING_H_
#define CLIO_CAE_CORE_HASHING_H_

#include <cstdint>
#include <cstdio>
#include <string>
#include <string_view>

namespace clio::cae::core::hashing {

/** FNV-1a 64-bit raw value over a byte range. */
inline uint64_t Fnv1a64(const void *data, size_t len) {
  const uint8_t *bytes = static_cast<const uint8_t *>(data);
  uint64_t h = 0xcbf29ce484222325ULL;       // FNV offset basis
  for (size_t i = 0; i < len; ++i) {
    h ^= bytes[i];
    h *= 0x100000001b3ULL;                  // FNV prime
  }
  return h;
}

/** FNV-1a 64-bit over a string_view, returned as 16-char lowercase hex. */
inline std::string Fnv1a64Hex(std::string_view s) {
  uint64_t h = Fnv1a64(s.data(), s.size());
  char buf[17];
  std::snprintf(buf, sizeof(buf), "%016llx",
                static_cast<unsigned long long>(h));
  return std::string(buf, 16);
}

/** Combine an existing hex digest with another value. Idiomatic chaining:
 *
 *     auto h = Fnv1a64Hex(input1);
 *     h = ChainHex(h, input2);
 *     h = ChainHex(h, std::to_string(version));
 *
 * Produces a deterministic order-sensitive composite hash. */
inline std::string ChainHex(std::string_view prev_hex, std::string_view next) {
  std::string combined;
  combined.reserve(prev_hex.size() + 1 + next.size());
  combined.append(prev_hex);
  combined.push_back('|');
  combined.append(next);
  return Fnv1a64Hex(combined);
}

}  // namespace clio::cae::core::hashing

#endif  // CLIO_CAE_CORE_HASHING_H_
