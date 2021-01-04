/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

/**
 * @file writer_impl.hpp
 * @brief cuDF-IO Parquet writer class implementation header
 */

#pragma once

#include <io/parquet/parquet.hpp>
#include <io/parquet/parquet_gpu.hpp>

#include <cudf/io/data_sink.hpp>
#include <io/utilities/hostdevice_vector.hpp>

#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/io/detail/parquet.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <memory>
#include <string>
#include <vector>

namespace cudf {
namespace io {
namespace detail {
namespace parquet {
// Forward internal classes
class parquet_column_view;

using namespace cudf::io::parquet;
using namespace cudf::io;

/**
 * @brief Chunked writer state struct. Contains pieces of information
 *        needed that span the write() / end() call process.
 */
struct pq_chunked_state {
  /// current write position for rowgroups/chunks
  std::size_t current_chunk_offset;
  /// only used in the write_chunked() case. copied from the (optionally) user supplied
  /// argument to write()
  bool single_write_mode;

  pq_chunked_state() = default;

  pq_chunked_state(SingleWriteMode mode = SingleWriteMode::NO)
    : single_write_mode({mode == SingleWriteMode::YES})
  {
  }
};

/**
 * @brief Implementation for parquet writer
 */
class writer::impl {
  // Parquet datasets are divided into fixed-size, independent rowgroups
  static constexpr uint32_t DEFAULT_ROWGROUP_MAXSIZE = 128 * 1024 * 1024;  // 128MB
  static constexpr uint32_t DEFAULT_ROWGROUP_MAXROWS = 1000000;            // Or at most 1M rows

  // rowgroups are divided into pages
  static constexpr uint32_t DEFAULT_TARGET_PAGE_SIZE = 512 * 1024;

 public:
  /**
   * @brief Constructor with writer options.
   *
   * @param filepath Filepath if storing dataset to a file
   * @param options Settings for controlling behavior
   * @param mr Device memory resource to use for device memory allocation
   */
  explicit impl(std::unique_ptr<data_sink> sink,
                parquet_writer_options const& options,
                rmm::mr::device_memory_resource* mr);

  /**
   * @brief Constructor with chunked writer options.
   *
   * @param filepath Filepath if storing dataset to a file
   * @param options Settings for controlling behavior
   * @param mr Device memory resource to use for device memory allocation
   **/
  explicit impl(std::unique_ptr<data_sink> sink,
                chunked_parquet_writer_options const& options,
                rmm::mr::device_memory_resource* mr);

  /**
   * @brief Destructor to complete any incomplete write and release resources.
   **/
  ~impl();

  /**
   * @brief Initializes the states before writing.
   *
   * @param[in] mode Option to write at once or in chunks.
   */
  void init_state(SingleWriteMode mode = SingleWriteMode::NO);

  /**
   * @brief Write an entire dataset to parquet format.
   *
   * @param table The set of columns
   * @param return_filemetadata If true, return the raw parquet file metadata
   * @param column_chunks_file_path Column chunks file path to be set in the raw output metadata
   * @param stream CUDA stream used for device memory operations and kernel launches.
   * @return unique_ptr to FileMetadata thrift message if requested
   */
  std::unique_ptr<std::vector<uint8_t>> write(table_view const& table,
                                              bool return_filemetadata,
                                              const std::string& column_chunks_file_path,
                                              rmm::cuda_stream_view stream);

  /**
   * @brief Writes a single subtable as part of a larger parquet file/table write,
   * normally used for chunked writing.
   *
   * @param[in] table The table information to be written
   * @param[in] mode Option to write at once or in chunks.
   * boundaries.
   */
  void write(table_view const& table, SingleWriteMode mode = SingleWriteMode::NO);

  /**
   * @brief Finishes the chunked/streamed write process.
   *
   * @param[in] return_filemetadata If true, return the raw parquet file metadata
   * @param[in] column_chunks_file_path Column chunks file path to be set in the raw output metadata
   * @return unique_ptr to FileMetadata thrift message if requested
   */
  std::unique_ptr<std::vector<uint8_t>> write_end(bool return_filemetadata = false,
                                                  const std::string& column_chunks_file_path = "");

 private:
  /**
   * @brief Gather page fragments
   *
   * @param frag Destination page fragments
   * @param col_desc column description array
   * @param num_columns Total number of columns
   * @param num_fragments Total number of fragments per column
   * @param num_rows Total number of rows
   * @param fragment_size Number of rows per fragment
   * @param stream CUDA stream used for device memory operations and kernel launches.
   */
  void init_page_fragments(hostdevice_vector<gpu::PageFragment>& frag,
                           hostdevice_vector<gpu::EncColumnDesc>& col_desc,
                           uint32_t num_columns,
                           uint32_t num_fragments,
                           uint32_t num_rows,
                           uint32_t fragment_size,
                           rmm::cuda_stream_view stream);
  /**
   * @brief Gather per-fragment statistics
   *
   * @param dst_stats output statistics
   * @param frag Input page fragments
   * @param col_desc column description array
   * @param num_columns Total number of columns
   * @param num_fragments Total number of fragments per column
   * @param fragment_size Number of rows per fragment
   * @param stream CUDA stream used for device memory operations and kernel launches.
   */
  void gather_fragment_statistics(statistics_chunk* dst_stats,
                                  hostdevice_vector<gpu::PageFragment>& frag,
                                  hostdevice_vector<gpu::EncColumnDesc>& col_desc,
                                  uint32_t num_columns,
                                  uint32_t num_fragments,
                                  uint32_t fragment_size,
                                  rmm::cuda_stream_view stream);
  /**
   * @brief Build per-chunk dictionaries and count data pages
   *
   * @param chunks column chunk array
   * @param col_desc column description array
   * @param num_rowgroups Total number of rowgroups
   * @param num_columns Total number of columns
   * @param num_dictionaries Total number of dictionaries
   * @param stream CUDA stream used for device memory operations and kernel launches.
   */
  void build_chunk_dictionaries(hostdevice_vector<gpu::EncColumnChunk>& chunks,
                                hostdevice_vector<gpu::EncColumnDesc>& col_desc,
                                uint32_t num_rowgroups,
                                uint32_t num_columns,
                                uint32_t num_dictionaries,
                                rmm::cuda_stream_view stream);
  /**
   * @brief Initialize encoder pages
   *
   * @param chunks column chunk array
   * @param col_desc column description array
   * @param pages encoder pages array
   * @param num_rowgroups Total number of rowgroups
   * @param num_columns Total number of columns
   * @param num_pages Total number of pages
   * @param num_stats_bfr Number of statistics buffers
   * @param stream CUDA stream used for device memory operations and kernel launches.
   */
  void init_encoder_pages(hostdevice_vector<gpu::EncColumnChunk>& chunks,
                          hostdevice_vector<gpu::EncColumnDesc>& col_desc,
                          gpu::EncPage* pages,
                          statistics_chunk* page_stats,
                          statistics_chunk* frag_stats,
                          uint32_t num_rowgroups,
                          uint32_t num_columns,
                          uint32_t num_pages,
                          uint32_t num_stats_bfr,
                          rmm::cuda_stream_view stream);
  /**
   * @brief Encode a batch pages
   *
   * @param chunks column chunk array
   * @param pages encoder pages array
   * @param num_columns Total number of columns
   * @param pages_in_batch number of pages in this batch
   * @param first_page_in_batch first page in batch
   * @param rowgroups_in_batch number of rowgroups in this batch
   * @param first_rowgroup first rowgroup in batch
   * @param comp_in compressor input array
   * @param comp_out compressor status array
   * @param page_stats optional page-level statistics (nullptr if none)
   * @param chunk_stats optional chunk-level statistics (nullptr if none)
   * @param stream CUDA stream used for device memory operations and kernel launches.
   */
  void encode_pages(hostdevice_vector<gpu::EncColumnChunk>& chunks,
                    gpu::EncPage* pages,
                    uint32_t num_columns,
                    uint32_t pages_in_batch,
                    uint32_t first_page_in_batch,
                    uint32_t rowgroups_in_batch,
                    uint32_t first_rowgroup,
                    gpu_inflate_input_s* comp_in,
                    gpu_inflate_status_s* comp_out,
                    const statistics_chunk* page_stats,
                    const statistics_chunk* chunk_stats,
                    rmm::cuda_stream_view stream);

 private:
  // TODO : figure out if we want to keep this. It is currently unused.
  rmm::mr::device_memory_resource* _mr = nullptr;

  size_t max_rowgroup_size_          = DEFAULT_ROWGROUP_MAXSIZE;
  size_t max_rowgroup_rows_          = DEFAULT_ROWGROUP_MAXROWS;
  size_t target_page_size_           = DEFAULT_TARGET_PAGE_SIZE;
  Compression compression_           = Compression::UNCOMPRESSED;
  statistics_freq stats_granularity_ = statistics_freq::STATISTICS_NONE;
  bool int96_timestamps              = false;
  // Cuda stream to be used
  rmm::cuda_stream_view stream_ = rmm::cuda_stream_default;
  // Overall file metadata.  Filled in during the process and written during write_chunked_end()
  cudf::io::parquet::FileMetaData md;
  // optional user metadata
  table_metadata_with_nullability user_metadata_with_nullability;
  // special parameter only used by detail::write() to indicate that we are guaranteeing
  // a single table write.  this enables some internal optimizations.
  table_metadata const* user_metadata = nullptr;
  // preserves chunked state
  std::unique_ptr<pq_chunked_state> state = nullptr;
  // to track if the output has been written to file
  bool is_written = false;

  std::vector<uint8_t> buffer_;
  std::unique_ptr<data_sink> out_sink_;
};

}  // namespace parquet
}  // namespace detail
}  // namespace io
}  // namespace cudf
