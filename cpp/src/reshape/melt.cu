/* Copyright 2018 NVIDIA Corporation.  All rights reserved. */

#include "dataframe/cudf_table.cuh"
#include <cudf.h>

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <vector>

void gdf_melt_length(size_t num_id_cols, size_t num_value_cols, size_t old_rows,
    size_t * new_rows, size_t * new_cols)
{
    *new_rows = old_rows * num_value_cols;
    *new_cols = num_id_cols + 2;
}

gdf_error gdf_melt(gdf_column *id_columns[], size_t num_id_cols, gdf_column *value_columns[], size_t num_value_cols,
    gdf_column *out_columns[])
{
    // Calculate dimensions
    gdf_size_type old_length = 0;
    if (num_id_cols != 0)
    {
        old_length = id_columns[0]->size;
    }
    else if (num_value_cols != 0)
    {
        old_length = value_columns[0]->size;
    }
    else
    {
        return GDF_DATASET_EMPTY;
    }
    gdf_size_type new_length = num_value_cols * old_length;

    // Step 1: tile id_columns
    size_t idx; // persist this outside loop to use for indexing out_columns
    for(idx = 0; idx < num_id_cols; idx++)
    {
        // step 1.1: make vector of pointers to same id_col
        auto cols_to_concat = std::vector<gdf_column *>(num_value_cols, id_columns[idx]);
        // step 1.2: Concat these into one output column
        gdf_error result = gdf_column_concat(out_columns[idx], cols_to_concat.data(), cols_to_concat.size());
        if (GDF_SUCCESS != result) return result;
    }

    // Step 2: Add variable column
    // num_value_cols number of categories
    // repeat each category old_length times.
    void * out_var_col_data = out_columns[idx]->data;
    auto make_category_column = [=] __device__ (gdf_size_type i)
    {
        int8_t * data = reinterpret_cast<decltype(data)> (out_var_col_data);
        data[i] = i / old_length;
    };
    thrust::for_each(thrust::counting_iterator<int>(0),
                    thrust::counting_iterator<int>(new_length),
                    make_category_column);
    CUDA_TRY( cudaGetLastError() );
    idx++;

    // Step 3: Add values column
    gdf_error result = gdf_column_concat(out_columns[idx], value_columns, num_value_cols);
    if (GDF_SUCCESS != result) return result;

    return GDF_SUCCESS;
}