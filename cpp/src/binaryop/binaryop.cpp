/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Copyright 2018-2019 BlazingDB, Inc.
 *     Copyright 2018 Christian Noboa Mardini <christian@blazingdb.com>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <bitmask/legacy/bitmask_ops.hpp> // remove
#include <cudf/utilities/error.hpp>
#include <utilities/cudf_utils.h> // remove
#include <cudf/cudf.h> // remove
#include <bitmask/legacy/legacy_bitmask.hpp> //remove
#include <cudf/legacy/copying.hpp> // remove/replace
#include <cudf/utilities/error.hpp> // wtf duplicate
#include <cudf/utilities/traits.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/detail/binaryop.hpp>
#include <cudf/binaryop.hpp>

#include <jit/launcher.h>
#include <jit/type.h>
#include <jit/parser.h>
#include <binaryop/jit/code/code.h>
#include <binaryop/jit/util.hpp>
#include <cudf/datetime.hpp> // replace eventually

#include <types.h.jit>
#include <types.hpp.jit>

namespace cudf {
namespace experimental {

namespace binops {

namespace jit {

  const std::string hash = "prog_binop";

  const std::vector<std::string> compiler_flags { "-std=c++14" };
  const std::vector<std::string> headers_name
        { "operation.h" , "traits.h", cudf_types_h, cudf_types_hpp };
  
  std::istream* headers_code(std::string filename, std::iostream& stream) {
      if (filename == "operation.h") {
          stream << code::operation;
          return &stream;
      }
      if (filename == "traits.h") {
          stream << code::traits;
          return &stream;
      }
      return nullptr;
  }

void binary_operation(mutable_column_view& out,
                      scalar const& lhs,
                      column_view const& rhs,
                      binary_operator ope,
                      cudaStream_t stream) {
  
  cudf::jit::launcher(
    hash, code::kernel, headers_name, compiler_flags, headers_code, stream
  ).set_kernel_inst(
    "kernel_v_s", // name of the kernel we are launching
    { cudf::jit::get_type_name(out.type()), // list of template arguments
      cudf::jit::get_type_name(rhs.type()),
      cudf::jit::get_type_name(lhs.type()),
      get_operator_name(ope, OperatorType::Reverse) } 
  ).launch(
    out.size(),
    cudf::jit::get_data_ptr(out),
    cudf::jit::get_data_ptr(rhs),
    cudf::jit::get_data_ptr(lhs)
  );

}

void binary_operation(mutable_column_view& out,
                      column_view const& lhs,
                      scalar const& rhs,
                      binary_operator ope,
                      cudaStream_t stream) {
  
  cudf::jit::launcher(
    hash, code::kernel, headers_name, compiler_flags, headers_code, stream
  ).set_kernel_inst(
    "kernel_v_s", // name of the kernel we are launching
    { cudf::jit::get_type_name(out.type()), // list of template arguments
      cudf::jit::get_type_name(lhs.type()),
      cudf::jit::get_type_name(rhs.type()),
      get_operator_name(ope, OperatorType::Direct) } 
  ).launch(
    out.size(),
    cudf::jit::get_data_ptr(out),
    cudf::jit::get_data_ptr(lhs),
    cudf::jit::get_data_ptr(rhs)
  );

}

void binary_operation(mutable_column_view& out,
                      column_view const& lhs,
                      column_view const& rhs,
                      binary_operator ope,
                      cudaStream_t stream) {

  cudf::jit::launcher(
    hash, code::kernel, headers_name, compiler_flags, headers_code, stream
  ).set_kernel_inst(
    "kernel_v_v", // name of the kernel we are launching
    { cudf::jit::get_type_name(out.type()), // list of template arguments
      cudf::jit::get_type_name(lhs.type()),
      cudf::jit::get_type_name(rhs.type()),
      get_operator_name(ope, OperatorType::Direct) } 
  ).launch(
    out.size(),
    cudf::jit::get_data_ptr(out),
    cudf::jit::get_data_ptr(lhs),
    cudf::jit::get_data_ptr(rhs)
  );

}

void binary_operation(mutable_column_view& out,
                      column_view const& lhs,
                      column_view const& rhs,
                      const std::string& ptx,
                      cudaStream_t stream) {

  std::string const output_type_name = cudf::jit::get_type_name(out.type());

  std::string ptx_hash = 
    hash + "." + std::to_string(std::hash<std::string>{}(ptx + output_type_name)); 
  std::string cuda_source = "\n#include <cudf/types.hpp>\n" +
    cudf::jit::parse_single_function_ptx(ptx, "GENERIC_BINARY_OP",
                                         output_type_name)
    + code::kernel;

  cudf::jit::launcher(
    ptx_hash, cuda_source, headers_name, compiler_flags, headers_code, stream
  ).set_kernel_inst(
    "kernel_v_v", // name of the kernel we are launching
    { output_type_name,                     // list of template arguments
      cudf::jit::get_type_name(lhs.type()),
      cudf::jit::get_type_name(rhs.type()),
      get_operator_name(GENERIC_BINARY, OperatorType::Direct) } 
  ).launch(
    out.size(),
    cudf::jit::get_data_ptr(out),
    cudf::jit::get_data_ptr(lhs),
    cudf::jit::get_data_ptr(rhs)
  );

}

}  // namespace jit
}  // namespace binops

namespace {
/**---------------------------------------------------------------------------*
 * @brief Computes output valid mask for op between a column and a scalar
 *
 * @param out_null_coun[out] number of nulls in output
 * @param valid_out preallocated output mask
 * @param valid_col input mask of column
 * @param valid_scalar bool indicating if scalar is valid
 * @param num_values number of values in input mask valid_col
 *---------------------------------------------------------------------------**/
auto scalar_col_valid_mask_and(column_view const& col, scalar const& s,
                               cudaStream_t stream,
                               rmm::mr::device_memory_resource *mr)
{
  if (col.size() == 0) {
    return rmm::device_buffer{};
  }

  if (not s.is_valid()) {
    return create_null_mask(col.size(), mask_state::ALL_NULL, stream, mr);
  } else if (s.is_valid() && col.nullable()) {
    return copy_bitmask(col, stream, mr);
  } else if (s.is_valid() && not col.nullable()) {
    return rmm::device_buffer{};
  }
}
} // namespace

namespace detail {

std::unique_ptr<column> binary_operation( scalar const& lhs,
                                          column_view const& rhs,
                                          binary_operator ope,
                                          data_type output_type,
                                          rmm::mr::device_memory_resource *mr,
                                          cudaStream_t stream)
{
  auto new_mask = scalar_col_valid_mask_and(rhs, lhs, stream, mr);
  auto out = make_numeric_column(output_type, rhs.size(), new_mask,
                                 cudf::UNKNOWN_NULL_COUNT, stream, mr);

  if (rhs.size() == 0)
    return out;

  auto out_view = out->mutable_view();
  binops::jit::binary_operation(out_view, lhs, rhs, ope, stream);
  return out;
}

std::unique_ptr<column> binary_operation( column_view const& lhs,
                                          scalar const& rhs,
                                          binary_operator ope,
                                          data_type output_type,
                                          rmm::mr::device_memory_resource *mr,
                                          cudaStream_t stream)
{
  auto new_mask = scalar_col_valid_mask_and(lhs, rhs, stream, mr);
  auto out = make_numeric_column(output_type, lhs.size(), new_mask,
                                 cudf::UNKNOWN_NULL_COUNT, stream, mr);

  if (lhs.size() == 0)
    return out;

  auto out_view = out->mutable_view();
  binops::jit::binary_operation(out_view, lhs, rhs, ope, stream);
  return out;  
}

std::unique_ptr<column> binary_operation( column_view const& lhs,
                                          column_view const& rhs,
                                          binary_operator ope,
                                          data_type output_type,
                                          rmm::mr::device_memory_resource *mr,
                                          cudaStream_t stream)
{
  CUDF_EXPECTS((lhs.size() == rhs.size()), "Column sizes don't match");

  // Check for 0 sized data
  if (lhs.size() == 0) // also rhs.size() == 0
    return make_numeric_column(output_type, 0);

  auto new_mask = bitmask_and(lhs, rhs, stream, mr);
  auto out = make_numeric_column(output_type, lhs.size(), new_mask,
                                 cudf::UNKNOWN_NULL_COUNT, stream, mr);

  // TODO: This whole shebang should be replaced by jitified header of chrono
        // gdf_column lhs_tmp{};
        // gdf_column rhs_tmp{};
        // // If the columns are GDF_DATE64 or timestamps with different time resolutions,
        // // cast the least-granular column to the other's resolution before the binop
        // std::tie(lhs_tmp, rhs_tmp) = cudf::datetime::cast_to_common_resolution(*lhs, *rhs);

        // if (lhs_tmp.size > 0) { lhs = &lhs_tmp; }
        // else if (rhs_tmp.size > 0) { rhs = &rhs_tmp; }

  auto out_view = out->mutable_view();
  binops::jit::binary_operation(out_view, lhs, rhs, ope, stream);
  return out;

  // gdf_column_free(&lhs_tmp);
  // gdf_column_free(&rhs_tmp);
}

std::unique_ptr<column> binary_operation( column_view const& lhs,
                                          column_view const& rhs,
                                          std::string const& ptx,
                                          data_type output_type,
                                          rmm::mr::device_memory_resource *mr,
                                          cudaStream_t stream)
{
  CUDF_EXPECTS((lhs.size() == rhs.size()), "Column sizes don't match");

  // Check for 0 sized data
  if (lhs.size() == 0) // also rhs.size() == 0
    return make_numeric_column(output_type, 0);

  auto new_mask = bitmask_and(lhs, rhs, stream, mr);
  auto out = make_numeric_column(output_type, lhs.size(), new_mask,
                                 cudf::UNKNOWN_NULL_COUNT, stream, mr);
  
  // Check for datatype
  auto is_type_supported = [] (data_type type) -> bool {
    return is_numeric(type) and 
           type.id() != type_id::INT8 and
           type.id() != type_id::BOOL8;
  };
  CUDF_EXPECTS(is_type_supported(lhs.type()), "Invalid/Unsupported lhs datatype");
  CUDF_EXPECTS(is_type_supported(rhs.type()), "Invalid/Unsupported rhs datatype");
  CUDF_EXPECTS(is_type_supported(output_type), "Invalid/Unsupported output datatype");
  
  auto out_view = out->mutable_view();
  binops::jit::binary_operation(out_view, lhs, rhs, ptx, stream);
  return out;
}

} // namespace detail

std::unique_ptr<column> binary_operation( scalar const& lhs,
                                          column_view const& rhs,
                                          binary_operator ope,
                                          data_type output_type,
                                          rmm::mr::device_memory_resource *mr)
{
  return detail::binary_operation(lhs, rhs, ope, output_type, mr);
}

std::unique_ptr<column> binary_operation( column_view const& lhs,
                                          scalar const& rhs,
                                          binary_operator ope,
                                          data_type output_type,
                                          rmm::mr::device_memory_resource *mr)
{
  return detail::binary_operation(lhs, rhs, ope, output_type, mr);
}

std::unique_ptr<column> binary_operation( column_view const& lhs,
                                          column_view const& rhs,
                                          binary_operator ope,
                                          data_type output_type,
                                          rmm::mr::device_memory_resource *mr)
{
  return detail::binary_operation(lhs, rhs, ope, output_type, mr);
}

std::unique_ptr<column> binary_operation(column_view const& lhs,
                                          column_view const& rhs,
                                          std::string const& ptx,
                                          data_type output_type,
                                          rmm::mr::device_memory_resource *mr)
{
  return detail::binary_operation(lhs, rhs, ptx, output_type, mr);
}

} // namespace experimental
} // namespace cudf