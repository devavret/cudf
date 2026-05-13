/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common.cuh"
#include "join/join_common_utils.cuh"

#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/row_operator/hashing.cuh>
#include <cudf/detail/row_operator/primitive_row_operators.cuh>
#include <cudf/hashing/detail/murmurhash3_x86_32.cuh>
#include <cudf/join/hash_join.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/type_checks.hpp>

#include <rmm/mr/polymorphic_allocator.hpp>

#include <cuda/iterator>

#include <limits>
#include <memory>

namespace cudf::detail {

bool is_trivial_join(table_view const& left, table_view const& right, join_kind join_type)
{
  if (left.is_empty() || right.is_empty()) { return true; }
  if ((join_kind::LEFT_JOIN == join_type) && (0 == left.num_rows())) { return true; }
  if ((join_kind::INNER_JOIN == join_type) && ((0 == left.num_rows()) || (0 == right.num_rows()))) {
    return true;
  }
  if ((join_kind::LEFT_SEMI_JOIN == join_type) && (0 == right.num_rows())) { return true; }
  if ((join_kind::LEFT_SEMI_JOIN == join_type || join_kind::LEFT_ANTI_JOIN == join_type) &&
      (0 == left.num_rows())) {
    return true;
  }
  return false;
}

void validate_hash_join_probe(table_view const& build, table_view const& probe, bool has_nulls)
{
  CUDF_EXPECTS(0 != probe.num_columns(), "Hash join probe table is empty", std::invalid_argument);
  CUDF_EXPECTS(build.num_columns() == probe.num_columns(),
               "Mismatch in number of columns to be joined on",
               std::invalid_argument);
  CUDF_EXPECTS(has_nulls || !cudf::has_nested_nulls(probe),
               "Probe table has nulls while build table was not hashed with null check.",
               std::invalid_argument);
  CUDF_EXPECTS(cudf::have_same_types(build, probe),
               "Mismatch in joining column data types",
               cudf::data_type_error);
}

namespace {
void build_hash_join(
  cudf::table_view const& build,
  std::shared_ptr<detail::row::equality::preprocessed_table> const& preprocessed_build,
  cudf::detail::hash_table_t& hash_table,
  bool has_nested_nulls,
  null_equality nulls_equal,
  [[maybe_unused]] bitmask_type const* bitmask,
  rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(0 != build.num_columns(), "Selected build dataset is empty", std::invalid_argument);
  CUDF_EXPECTS(0 != build.num_rows(), "Build side table has no rows", std::invalid_argument);

  auto insert_rows = [&](auto const& build, auto const& d_hasher) {
    auto const iter = cudf::detail::make_counting_transform_iterator(0, pair_fn{d_hasher});

    if (nulls_equal == cudf::null_equality::EQUAL or not nullable(build)) {
      hash_table.insert(iter, iter + build.num_rows(), stream.value());
    } else {
      auto const stencil = cuda::counting_iterator<size_type>{0};
      auto const pred    = row_is_valid{bitmask};

      hash_table.insert_if(iter, iter + build.num_rows(), stencil, pred, stream.value());
    }
  };

  auto const nulls = nullate::DYNAMIC{has_nested_nulls};

  if (cudf::detail::is_primitive_row_op_compatible(build)) {
    auto const d_hasher = cudf::detail::row::primitive::row_hasher{nulls, preprocessed_build};

    insert_rows(build, d_hasher);
  } else {
    auto const row_hash = detail::row::hash::row_hasher{preprocessed_build};
    auto const d_hasher = row_hash.device_hasher(nulls);

    insert_rows(build, d_hasher);
  }
}
}  // namespace

template <typename Hasher>
hash_join<Hasher>::hash_join(cudf::table_view const& build,
                             bool has_nulls,
                             cudf::null_equality compare_nulls,
                             rmm::cuda_stream_view stream)
  : hash_join{build, has_nulls, compare_nulls, CUCO_DESIRED_LOAD_FACTOR, stream}
{
}

template <typename Hasher>
hash_join<Hasher>::hash_join(cudf::table_view const& build,
                             bool has_nulls,
                             cudf::null_equality compare_nulls,
                             double load_factor,
                             rmm::cuda_stream_view stream)
  : _has_nulls(has_nulls),
    _is_empty{build.num_rows() == 0},
    _nulls_equal{compare_nulls},
    _load_factor{load_factor},
    _impl{std::make_unique<impl>(impl{typename impl::hash_table_t{
      cuco::extent{static_cast<size_t>(build.num_rows())},
      load_factor,
      cuco::empty_key{cuco::pair{std::numeric_limits<hash_value_type>::max(), cudf::JoinNoMatch}},
      {},
      {},
      {},
      {},
      rmm::mr::polymorphic_allocator<char>{},
      stream.value()}})},
    _build{build},
    _preprocessed_build{cudf::detail::row::equality::preprocessed_table::create(_build, stream)}
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(0 != build.num_columns(), "Hash join build table is empty", std::invalid_argument);
  CUDF_EXPECTS(load_factor > 0 && load_factor <= 1,
               "Invalid load factor: must be greater than 0 and less than or equal to 1.",
               std::invalid_argument);

  if (_is_empty) { return; }

  auto const row_bitmask =
    cudf::detail::bitmask_and(build, stream, cudf::get_current_device_resource_ref()).first;
  cudf::detail::build_hash_join(_build,
                                _preprocessed_build,
                                _impl->_hash_table,
                                _has_nulls,
                                _nulls_equal,
                                reinterpret_cast<bitmask_type const*>(row_bitmask.data()),
                                stream);
}

template <typename Hasher>
hash_join<Hasher>::hash_join(cudf::table_view const& build,
                             bool has_nulls,
                             cudf::null_equality compare_nulls,
                             double load_factor,
                             rmm::cuda_stream_view stream,
                             no_insert_t)
  : _has_nulls(has_nulls),
    _is_empty{build.num_rows() == 0},
    _nulls_equal{compare_nulls},
    _load_factor{load_factor},
    _impl{std::make_unique<impl>(impl{typename impl::hash_table_t{
      cuco::extent{static_cast<size_t>(build.num_rows())},
      load_factor,
      cuco::empty_key{cuco::pair{std::numeric_limits<hash_value_type>::max(), cudf::JoinNoMatch}},
      {},
      {},
      {},
      {},
      rmm::mr::polymorphic_allocator<char>{},
      stream.value()}})},
    _build{build},
    _preprocessed_build{cudf::detail::row::equality::preprocessed_table::create(_build, stream)}
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(0 != build.num_columns(), "Hash join build table is empty", std::invalid_argument);
  CUDF_EXPECTS(load_factor > 0 && load_factor <= 1,
               "Invalid load factor: must be greater than 0 and less than or equal to 1.",
               std::invalid_argument);
  // Intentionally skip the build_hash_join insert kernel. Slots remain in
  // cuco's empty-sentinel state until apply_storage runs.
}

template <typename Hasher>
cudf::hash_join_storage hash_join<Hasher>::release_storage(rmm::cuda_stream_view stream) const
{
  CUDF_FUNC_RANGE();
  auto& table         = _impl->_hash_table;
  auto const slots    = table.capacity();
  using slot_type     = std::remove_pointer_t<decltype(table.data())>;
  auto const slot_sz  = sizeof(slot_type);
  auto const total_sz = slots * slot_sz;

  cudf::hash_join_storage out;
  out.slots         = rmm::device_buffer{total_sz, stream};
  out.slot_count    = slots;
  out.slot_bytes    = slot_sz;
  out.compare_nulls = _nulls_equal;
  out.has_nulls     = _has_nulls;
  out.load_factor   = _load_factor;
  CUDF_CUDA_TRY(cudaMemcpyAsync(
    out.slots.data(), table.data(), total_sz, cudaMemcpyDeviceToDevice, stream.value()));
  return out;
}

template <typename Hasher>
void hash_join<Hasher>::apply_storage(cudf::hash_join_storage const& storage,
                                      rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();
  auto& table        = _impl->_hash_table;
  auto const slots   = table.capacity();
  using slot_type    = std::remove_pointer_t<decltype(table.data())>;
  auto const slot_sz = sizeof(slot_type);
  CUDF_EXPECTS(storage.slot_count == slots,
               "hash_join_storage slot_count does not match target hash table capacity",
               std::invalid_argument);
  CUDF_EXPECTS(storage.slot_bytes == slot_sz,
               "hash_join_storage slot_bytes does not match target slot layout",
               std::invalid_argument);
  CUDF_CUDA_TRY(cudaMemcpyAsync(
    table.data(), storage.slots.data(), slots * slot_sz, cudaMemcpyDeviceToDevice, stream.value()));
}

template hash_join<hash_join_hasher>::hash_join(cudf::table_view const& build,
                                                bool has_nulls,
                                                cudf::null_equality compare_nulls,
                                                rmm::cuda_stream_view stream);

template hash_join<hash_join_hasher>::hash_join(cudf::table_view const& build,
                                                bool has_nulls,
                                                cudf::null_equality compare_nulls,
                                                double load_factor,
                                                rmm::cuda_stream_view stream);

template hash_join<hash_join_hasher>::hash_join(cudf::table_view const& build,
                                                bool has_nulls,
                                                cudf::null_equality compare_nulls,
                                                double load_factor,
                                                rmm::cuda_stream_view stream,
                                                hash_join<hash_join_hasher>::no_insert_t);

template cudf::hash_join_storage hash_join<hash_join_hasher>::release_storage(
  rmm::cuda_stream_view stream) const;

template void hash_join<hash_join_hasher>::apply_storage(cudf::hash_join_storage const& storage,
                                                        rmm::cuda_stream_view stream);

template <typename Hasher>
hash_join<Hasher>::~hash_join() = default;

template hash_join<hash_join_hasher>::~hash_join();

}  // namespace cudf::detail

namespace cudf {

hash_join::~hash_join() = default;

hash_join::hash_join(cudf::table_view const& build,
                     null_equality compare_nulls,
                     rmm::cuda_stream_view stream)
  : hash_join(
      build, nullable_join::YES, compare_nulls, cudf::detail::CUCO_DESIRED_LOAD_FACTOR, stream)
{
}

hash_join::hash_join(cudf::table_view const& build,
                     nullable_join has_nulls,
                     null_equality compare_nulls,
                     double load_factor,
                     rmm::cuda_stream_view stream)
  : _impl{std::make_unique<impl_type>(
      build, has_nulls == nullable_join::YES, compare_nulls, load_factor, stream)}
{
}

hash_join::hash_join(std::unique_ptr<impl_type> impl) : _impl{std::move(impl)} {}

hash_join_storage hash_join::release_storage(rmm::cuda_stream_view stream) const
{
  return _impl->release_storage(stream);
}

std::unique_ptr<hash_join> hash_join::from_storage(hash_join_storage storage,
                                                   cudf::table_view const& build,
                                                   rmm::cuda_stream_view stream)
{
  auto const has_nulls     = storage.has_nulls;
  auto const compare_nulls = storage.compare_nulls;
  auto const load_factor   = storage.load_factor;
  auto detail_impl =
    std::make_unique<impl_type>(build,
                                has_nulls,
                                compare_nulls,
                                load_factor,
                                stream,
                                typename impl_type::no_insert_t{});
  detail_impl->apply_storage(storage, stream);
  // Use unique_ptr<hash_join>{new hash_join(...)} because hash_join is
  // non-movable: std::make_unique would still work, but we keep the
  // private constructor explicit at the call site.
  return std::unique_ptr<hash_join>{new hash_join(std::move(detail_impl))};
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join::inner_join(cudf::table_view const& probe,
                      std::optional<std::size_t> output_size,
                      rmm::cuda_stream_view stream,
                      rmm::device_async_resource_ref mr) const
{
  return _impl->inner_join(probe, output_size, stream, mr);
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join::left_join(cudf::table_view const& probe,
                     std::optional<std::size_t> output_size,
                     rmm::cuda_stream_view stream,
                     rmm::device_async_resource_ref mr) const
{
  return _impl->left_join(probe, output_size, stream, mr);
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join::full_join(cudf::table_view const& probe,
                     std::optional<std::size_t> output_size,
                     rmm::cuda_stream_view stream,
                     rmm::device_async_resource_ref mr) const
{
  return _impl->full_join(probe, output_size, stream, mr);
}

std::size_t hash_join::inner_join_size(cudf::table_view const& probe,
                                       rmm::cuda_stream_view stream) const
{
  return _impl->inner_join_size(probe, stream);
}

std::size_t hash_join::left_join_size(cudf::table_view const& probe,
                                      rmm::cuda_stream_view stream) const
{
  return _impl->left_join_size(probe, stream);
}

std::size_t hash_join::full_join_size(cudf::table_view const& probe,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr) const
{
  return _impl->full_join_size(probe, stream, mr);
}

cudf::join_match_context hash_join::inner_join_match_context(
  cudf::table_view const& probe,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  return _impl->inner_join_match_context(probe, stream, mr);
}

cudf::join_match_context hash_join::left_join_match_context(cudf::table_view const& probe,
                                                            rmm::cuda_stream_view stream,
                                                            rmm::device_async_resource_ref mr) const
{
  return _impl->left_join_match_context(probe, stream, mr);
}

cudf::join_match_context hash_join::full_join_match_context(cudf::table_view const& probe,
                                                            rmm::cuda_stream_view stream,
                                                            rmm::device_async_resource_ref mr) const
{
  return _impl->full_join_match_context(probe, stream, mr);
}

}  // namespace cudf
