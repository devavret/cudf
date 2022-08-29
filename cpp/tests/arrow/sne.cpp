#include "cudf/binaryop.hpp"
#include "cudf/column/column_factories.hpp"
#include "cudf/transform.hpp"
#include "cudf_test/column_utilities.hpp"
#include <cudf/column/column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf_test/base_fixture.hpp>

#include <rmm/device_buffer.hpp>

#include <arrow/api.h>
#include <arrow/compute/api.h>
#include <arrow/compute/exec.h>
#include <arrow/compute/exec/exec_plan.h>
#include <arrow/result.h>

#include <cuda.h>
#include <iostream>

struct ArrowSNETest : public cudf::test::BaseFixture {
};

std::unique_ptr<cudf::column> column_from_array_span(arrow::ArraySpan const& arr)
{
  switch (arr.type->id()) {
    case arrow::Type::STRING: {
      // buffer 0 = nullmask
      // buffer 1 = offsets
      // buffer 2 = chars
      auto nullmask = (arr.buffers[0].data)
                        ? rmm::device_buffer(arr.buffers[0].data,
                                             cudf::bitmask_allocation_size_bytes(arr.length),
                                             rmm::cuda_stream_default)
                        : rmm::device_buffer();

      auto offset_span = cudf::host_span<int32_t const>{
        reinterpret_cast<int32_t*>(arr.buffers[1].data), arr.buffers[1].size / sizeof(int32_t)};

      auto d_off = cudf::detail::make_device_uvector_async(offset_span, rmm::cuda_stream_default);

      auto chars_span = cudf::host_span<char const>{reinterpret_cast<char*>(arr.buffers[2].data),
                                                    static_cast<size_t>(arr.buffers[2].size)};

      auto d_chars = cudf::detail::make_device_uvector_async(chars_span, rmm::cuda_stream_default);

      return cudf::make_strings_column(
        arr.length, std::move(d_off), std::move(d_chars), std::move(nullmask));
    }
    default: CUDF_FAIL("unsupported");
  }
}

namespace arrow::compute {

// -------------------------------------------------------------------------------------------------
// Scalar Compare

struct StringEqual {
  static Status ArrayArray(KernelContext* ctx,
                           const ArraySpan& arg0,
                           const ArraySpan& arg1,
                           ExecResult* out)
  {
    Status st = Status::OK();
    auto col0 = column_from_array_span(arg0);
    cudf::test::print(*col0);

    auto col1 = column_from_array_span(arg1);
    cudf::test::print(*col1);

    auto result = cudf::binary_operation(
      *col0, *col1, cudf::binary_operator::EQUAL, cudf::data_type(cudf::type_id::BOOL8));

    cudf::test::print(*result);

    auto result_bits = cudf::bools_to_mask(*result);

    cudaMemcpy(out->array_span()->buffers[1].data,
               result_bits.first->data(),
               result_bits.first->size(),
               cudaMemcpyDefault);

    return st;
  }

  static Status Exec(KernelContext* ctx, const ExecSpan& batch, ExecResult* out)
  {
    if (batch[0].is_array()) {
      if (batch[1].is_array()) { return ArrayArray(ctx, batch[0].array, batch[1].array, out); }
    }
    DCHECK(false);
    return Status::Invalid("Should be unreachable");
  }
};

struct CompareGPUFunction : public ScalarFunction {
  using ScalarFunction::ScalarFunction;
};

std::shared_ptr<ScalarFunction> MakeCompareFunction(std::string name, FunctionDoc doc)
{
  auto func = std::make_shared<CompareGPUFunction>(name, Arity::Binary(), std::move(doc));

  // for (const std::shared_ptr<DataType>& ty : NumericTypes()) {
  //   AddPrimitiveCompare<Op>(ty, func.get());
  // }

  for (const std::shared_ptr<DataType>& ty : BaseBinaryTypes()) {
    auto exec = StringEqual::Exec;
    DCHECK_OK(func->AddKernel({ty, ty}, boolean(), std::move(exec)));
  }

  return func;
}

const FunctionDoc equal_doc2{"Compare values for equality (x == y)",
                             ("A null on either side emits a null comparison result."),
                             {"x", "y"}};

}  // namespace arrow::compute

arrow::Status ExecutePlanAndCollectAsTable(
  arrow::compute::ExecContext& exec_context,
  std::shared_ptr<arrow::compute::ExecPlan> plan,
  std::shared_ptr<arrow::Schema> schema,
  arrow::AsyncGenerator<arrow::util::optional<arrow::compute::ExecBatch>> sink_gen)
{
  // translate sink_gen (async) to sink_reader (sync)
  std::shared_ptr<arrow::RecordBatchReader> sink_reader =
    arrow::compute::MakeGeneratorReader(schema, std::move(sink_gen), exec_context.memory_pool());

  // validate the ExecPlan
  ARROW_RETURN_NOT_OK(plan->Validate());
  std::cout << "ExecPlan created : " << plan->ToString() << std::endl;
  // start the ExecPlan
  ARROW_RETURN_NOT_OK(plan->StartProducing());

  // collect sink_reader into a Table
  std::shared_ptr<arrow::Table> response_table;

  ARROW_ASSIGN_OR_RAISE(response_table, arrow::Table::FromRecordBatchReader(sink_reader.get()));

  std::cout << "Results : " << std::endl << response_table->ToString() << std::endl;

  // stop producing
  plan->StopProducing();
  // plan mark finished
  auto future = plan->finished();
  return future.status();
}

arrow::Status Execute()
{
  std::cout << arrow::GetBuildInfo().version_string << std::endl;

  // auto null_long = std::numeric_limits<int>::quiet_NaN();

  arrow::Int32Builder int_builder;
  ARROW_RETURN_NOT_OK(int_builder.Append(5));
  ARROW_RETURN_NOT_OK(int_builder.Append(10));
  ARROW_RETURN_NOT_OK(int_builder.AppendNull());
  ARROW_RETURN_NOT_OK(int_builder.Append(20));
  ARROW_RETURN_NOT_OK(int_builder.Append(10));
  ARROW_RETURN_NOT_OK(int_builder.Append(10));
  ARROW_ASSIGN_OR_RAISE(std::shared_ptr<arrow::Array> int_array, int_builder.Finish());

  arrow::StringBuilder str_builder;
  ARROW_RETURN_NOT_OK(str_builder.Append("dog"));
  ARROW_RETURN_NOT_OK(str_builder.Append("cat"));
  ARROW_RETURN_NOT_OK(str_builder.Append("dog"));
  ARROW_RETURN_NOT_OK(str_builder.Append("bird"));
  ARROW_RETURN_NOT_OK(str_builder.Append("dog"));
  ARROW_RETURN_NOT_OK(str_builder.Append("dog"));
  ARROW_ASSIGN_OR_RAISE(std::shared_ptr<arrow::Array> str_array, str_builder.Finish());

  arrow::StringBuilder str_builder_2;
  ARROW_RETURN_NOT_OK(str_builder_2.Append("dog"));
  ARROW_RETURN_NOT_OK(str_builder_2.Append("dog"));
  ARROW_RETURN_NOT_OK(str_builder_2.Append("dog"));
  ARROW_RETURN_NOT_OK(str_builder_2.Append("dog"));
  ARROW_RETURN_NOT_OK(str_builder_2.Append("dog"));
  ARROW_RETURN_NOT_OK(str_builder_2.Append("dog"));
  ARROW_ASSIGN_OR_RAISE(std::shared_ptr<arrow::Array> str_array_2, str_builder_2.Finish());

  std::vector<std::shared_ptr<arrow::Field>> schema_vector = {arrow::field("int", arrow::int32()),
                                                              arrow::field("str", arrow::utf8()),
                                                              arrow::field("str2", arrow::utf8())};

  auto schema = std::make_shared<arrow::Schema>(schema_vector);

  std::shared_ptr<arrow::Table> table =
    arrow::Table::Make(schema, {int_array, str_array, str_array_2});

  std::cout << "Data : " << std::endl << table->ToString() << std::endl;

  arrow::compute::ExecContext exec_context;

  ARROW_ASSIGN_OR_RAISE(std::shared_ptr<arrow::compute::ExecPlan> plan,
                        arrow::compute::ExecPlan::Make(&exec_context));

  int max_batch_size        = 100;
  auto table_source_options = arrow::compute::TableSourceNodeOptions{table, max_batch_size};
  arrow::AsyncGenerator<arrow::util::optional<arrow::compute::ExecBatch>> sink_gen;

  // source node
  ARROW_ASSIGN_OR_RAISE(
    arrow::compute::ExecNode * source_node,
    arrow::compute::MakeExecNode("table_source", plan.get(), {}, table_source_options));

  DCHECK_OK(arrow::compute::GetFunctionRegistry()->AddFunction(
    arrow::compute::MakeCompareFunction("equal", arrow::compute::equal_doc2), true));

  // filter node
  arrow::compute::ExecNode* filter_node;
  ARROW_ASSIGN_OR_RAISE(filter_node,
                        arrow::compute::MakeExecNode(
                          "filter",
                          plan.get(),
                          {source_node},
                          arrow::compute::FilterNodeOptions{arrow::compute::equal(
                            arrow::compute::field_ref("str"), arrow::compute::field_ref("str2"))}));

  // sink node
  ARROW_RETURN_NOT_OK(arrow::compute::MakeExecNode(
    "sink", plan.get(), {filter_node}, arrow::compute::SinkNodeOptions{&sink_gen}));

  auto result =
    ExecutePlanAndCollectAsTable(exec_context, plan, filter_node->output_schema(), sink_gen);

  return arrow::Status::OK();
};

TEST_F(ArrowSNETest, FilterExpression)
{
  auto status = Execute();
  if (!status.ok()) { std::cerr << "Error occurred : " << status.message() << std::endl; }
}
