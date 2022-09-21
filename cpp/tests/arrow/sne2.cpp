#include "cudf/binaryop.hpp"
#include "cudf/column/column_factories.hpp"
#include "cudf/scalar/scalar_factories.hpp"
#include "cudf/stream_compaction.hpp"
#include "cudf/transform.hpp"
#include "cudf_test/column_utilities.hpp"
#include "cudf_test/column_wrapper.hpp"
#include <cudf/column/column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf_test/base_fixture.hpp>

#include <rmm/device_buffer.hpp>

#include <arrow/api.h>
#include <arrow/compute/api.h>
#include <arrow/compute/exec.h>
#include <arrow/compute/exec/exec_plan.h>
#include <arrow/compute/exec/expression.h>
#include <arrow/result.h>

#include <cuda.h>
#include <iostream>
#include <variant>

struct ArrowSNETest : public cudf::test::BaseFixture {
};

std::unique_ptr<cudf::column> column_from_array_span(arrow::ArraySpan const& arr)
{
  // buffer 0 = nullmask
  auto nullmask = (arr.buffers[0].data)
                    ? rmm::device_buffer(arr.buffers[0].data,
                                         cudf::bitmask_allocation_size_bytes(arr.length),
                                         rmm::cuda_stream_default)
                    : rmm::device_buffer();
  switch (arr.type->id()) {
    case arrow::Type::BOOL: {
      auto bool_bits =
        rmm::device_buffer(arr.buffers[1].data, arr.buffers[1].size, rmm::cuda_stream_default);
      auto bool_bytes =
        cudf::mask_to_bools(reinterpret_cast<cudf::bitmask_type*>(bool_bits.data()), 0, arr.length);
      bool_bytes->set_null_mask(std::move(nullmask));
      return bool_bytes;
    }
    case arrow::Type::INT32: {
      auto data =
        rmm::device_buffer(arr.buffers[1].data, arr.buffers[1].size, rmm::cuda_stream_default);
      return std::make_unique<cudf::column>(
        cudf::data_type(cudf::type_id::INT32), arr.length, std::move(data), std::move(nullmask));
    }
    case arrow::Type::STRING: {
      // buffer 1 = offsets
      // buffer 2 = chars
      auto offset_span = cudf::host_span<int32_t const>{
        reinterpret_cast<int32_t*>(arr.buffers[1].data), arr.buffers[1].size / sizeof(int32_t)};

      auto d_off = cudf::detail::make_device_uvector_async(offset_span, rmm::cuda_stream_default);

      auto chars_span = cudf::host_span<char const>{reinterpret_cast<char*>(arr.buffers[2].data),
                                                    static_cast<size_t>(arr.buffers[2].size)};

      auto d_chars = cudf::detail::make_device_uvector_async(chars_span, rmm::cuda_stream_default);

      return cudf::make_strings_column(
        arr.length, std::move(d_off), std::move(d_chars), std::move(nullmask));
    }
    default: CUDF_FAIL("unsupported arrow type");
  }
}

void copy_column_to_array_span(cudf::column_view const& col, arrow::ArraySpan* span)
{
  // copy bitmask
  if (col.nullable()) {
    cudaMemcpy(span->buffers[0].data,
               col.null_mask(),
               cudf::bitmask_allocation_size_bytes(col.size()),
               cudaMemcpyDefault);
  }
  switch (col.type().id()) {
    case cudf::type_id::INT32: {
      cudaMemcpy(span->buffers[1].data,
                 col.head(),
                 col.size() * cudf::size_of(col.type()),
                 cudaMemcpyDefault);
      break;
    }
    default: CUDF_FAIL("unsupported type");
  }
}

// struct get_scalar_from_literal {
//   template <typename T>
//   cudf::scalar operator()()
//   {
//   }
// };

// cudf::ast::expression expression_to_cudf(arrow::compute::Expression expression)
// {
//   // expression can be literal/field_ref/call. Map to literal/column_reference/operation
//   if (expression.literal()) {
//     // TODO (dm): can literal be column? Makes no sense since ExecBatch can be any size.
//     // TODO (dm): need type dispatching from arrow to libcudf types
//     // TODO (dm): No string literal support. Either PR to cudf or use binaryop workaround.
//     // auto scalar = cudf::numeric_scalar<typename T>;
//     return
// cudf::ast::literal(cudf::make_numeric_scalar(cudf::data_type(cudf::type_id::INT32)));
//   }
// }

std::variant<std::unique_ptr<cudf::column>, cudf::column_view> recursive_expr_eval(
  cudf::table_view const& table, arrow::compute::Expression const& expr)
{
  if (auto param = expr.parameter()) { return table.column(param->indices[0]); }

  if (auto call = expr.call()) {
    if (call->function_name == "equal") {
      CUDF_EXPECTS(call->arguments.size() == 2, "Binary comparison");
      auto lhs      = recursive_expr_eval(table, call->arguments[0]);
      auto rhs      = recursive_expr_eval(table, call->arguments[1]);
      auto lhs_view = std::get_if<cudf::column_view>(&lhs)
                        ? std::get<cudf::column_view>(lhs)
                        : std::get<std::unique_ptr<cudf::column>>(lhs)->view();
      auto rhs_view = std::get_if<cudf::column_view>(&rhs)
                        ? std::get<cudf::column_view>(rhs)
                        : std::get<std::unique_ptr<cudf::column>>(rhs)->view();
      return cudf::binary_operation(
        lhs_view, rhs_view, cudf::binary_operator::EQUAL, cudf::data_type(cudf::type_id::BOOL8));
    } else {
      CUDF_FAIL("Unsupported operation");
    }
  } else {
    CUDF_FAIL("Unsupported expression types");
  }
}

namespace arrow::compute {

namespace {

class GPUFilterNode : public MapNode {
 public:
  GPUFilterNode(ExecPlan* plan,
                std::vector<ExecNode*> inputs,
                std::shared_ptr<Schema> output_schema,
                Expression filter,
                bool async_mode)
    : MapNode(plan, std::move(inputs), std::move(output_schema), async_mode),
      filter_(std::move(filter))
  {
  }

  static Result<ExecNode*> Make(ExecPlan* plan,
                                std::vector<ExecNode*> inputs,
                                const ExecNodeOptions& options)
  {
    RETURN_NOT_OK(ValidateExecNodeInputs(plan, inputs, 1, "GPUFilterNode"));
    auto schema = inputs[0]->output_schema();

    const auto& filter_options = ::arrow::internal::checked_cast<const FilterNodeOptions&>(options);

    auto filter_expression = filter_options.filter_expression;
    if (!filter_expression.IsBound()) {
      ARROW_ASSIGN_OR_RAISE(filter_expression,
                            filter_expression.Bind(*schema, plan->exec_context()));
    }

    if (filter_expression.type()->id() != Type::BOOL) {
      return Status::TypeError("Filter expression must evaluate to bool, but ",
                               filter_expression.ToString(),
                               " evaluates to ",
                               filter_expression.type()->ToString());
    }
    return plan->EmplaceNode<GPUFilterNode>(plan,
                                            std::move(inputs),
                                            std::move(schema),
                                            std::move(filter_expression),
                                            filter_options.async_mode);
  }

  const char* kind_name() const override { return "GPUFilterNode"; }

  Result<ExecBatch> DoFilter(const ExecBatch& target)
  {
    ARROW_ASSIGN_OR_RAISE(Expression simplified_filter,
                          SimplifyWithGuarantee(filter_, target.guarantee));

    // Convert expression to cudf::ast
    // auto cudf_expr = expression_to_cudf(simplified_filter);

    // Get cudf::table_view of ExecBatch
    std::vector<std::unique_ptr<cudf::column>> columns;
    // TODO (dm): Handle scalars in batch, or ensure no scalars in batch
    std::transform(
      target.values.begin(), target.values.end(), std::back_inserter(columns), [](Datum const& c) {
        return column_from_array_span(*c.array());
      });
    auto table = cudf::table(std::move(columns));

    auto mask = recursive_expr_eval(table, simplified_filter);
    // auto mask = cudf::compute_column(table, cudf_expr);

    auto mask_view = std::get_if<cudf::column_view>(&mask)
                       ? std::get<cudf::column_view>(mask)
                       : std::get<std::unique_ptr<cudf::column>>(mask)->view();

    cudf::test::print(mask_view);

    auto output = cudf::apply_boolean_mask(table, mask_view);
    for (auto const& col : output->view()) {
      cudf::test::print(col);
    }

    // convert back to ExecBatch
    std::vector<Datum> values;

    return ExecBatch::Make(std::move(values));
  }

  void InputReceived(ExecNode* input, ExecBatch batch) override
  {
    DCHECK_EQ(input, inputs_[0]);
    auto func = [this](ExecBatch batch) {
      auto result = DoFilter(std::move(batch));
      return result;
    };
    this->SubmitTask(std::move(func), std::move(batch));
  }

 protected:
  std::string ToStringExtra(int indent = 0) const override
  {
    return "filter=" + filter_.ToString();
  }

 private:
  Expression filter_;
};

}  // namespace

namespace internal {
void RegisterGPUFilterNode(ExecFactoryRegistry* registry)
{
  DCHECK_OK(registry->AddFactory("gpu_filter", GPUFilterNode::Make));
}

}  // namespace internal

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

  arrow::compute::internal::RegisterGPUFilterNode(arrow::compute::default_exec_factory_registry());

  // source node
  ARROW_ASSIGN_OR_RAISE(
    arrow::compute::ExecNode * source_node,
    arrow::compute::MakeExecNode("table_source", plan.get(), {}, table_source_options));

  // filter node
  arrow::compute::ExecNode* filter_node;
  ARROW_ASSIGN_OR_RAISE(filter_node,
                        arrow::compute::MakeExecNode(
                          "gpu_filter",
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

// TEST_F(ArrowSNETest, DecimalAST)
// {
//   auto col =
//     cudf::test::fixed_point_column_wrapper<int32_t>({1090, 1505, 602}, numeric::scale_type(2));

//   cudf::ast::operation(cudf::ast::literal())
//   cudf::compute_column(cudf::table_view({col}), )
// }
