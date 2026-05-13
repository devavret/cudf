/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/types.hpp>
#include <cudf/utilities/export.hpp>

#include <rmm/device_buffer.hpp>

#include <cstddef>

namespace CUDF_EXPORT cudf {

/**
 * @brief Owned, device-resident snapshot of a `cudf::hash_join`'s internal
 *        hash-table slot storage.
 *
 * Produced by `cudf::hash_join::release_storage()` and consumed by
 * `cudf::hash_join::from_storage()`. The struct is movable and owns its
 * device buffer. Callers may stage `slots` to and from host memory between
 * release and restore; cuDF itself performs no host-device copies.
 */
struct hash_join_storage {
  /// Device buffer holding the raw bytes of the cuco multiset's slot
  /// storage. Size equals `slot_count * slot_bytes`.
  rmm::device_buffer slots;
  /// Number of slots in the underlying cuco bucket storage.
  std::size_t slot_count{0};
  /// Size in bytes of a single slot (cuco::pair<hash_value_type, size_type>).
  std::size_t slot_bytes{0};
  /// Whether nulls in join keys were treated as equal when this storage was
  /// built. Must match on restore.
  null_equality compare_nulls{null_equality::UNEQUAL};
  /// Whether the original build/probe inputs were declared as possibly
  /// containing nulls. Must match on restore.
  bool has_nulls{true};
  /// Load factor with which the cuco hash table was built. Must match on
  /// restore so the same capacity is re-allocated.
  double load_factor{0.5};
};

}  // namespace CUDF_EXPORT cudf
