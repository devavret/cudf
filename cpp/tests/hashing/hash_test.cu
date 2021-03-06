/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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

 #include <cstdlib>
 #include <iostream>
 #include <vector>
 
 #include <thrust/device_vector.h>
 
 #include "gtest/gtest.h"
 #include "tests/utilities/cudf_test_fixtures.h"

#include <cudf.h>
 #include <cudf/functions.h>
 #include <rmm/thrust_rmm_allocator.h>
 
 // thrust::device_vector set to use rmmAlloc and rmmFree.
 template <typename T>
 using Vector = thrust::device_vector<T, rmm_allocator<T>>;

 struct gdf_hashing_test : public GdfTest {};
 
 TEST_F(gdf_hashing_test, allDtypesTest) {
 
	 int nrows = 5;
	 int ncols = 6;
 
	 gdf_column **inputCol;
	 inputCol = (gdf_column**) malloc(sizeof(gdf_column*)* ncols);
	 for(int i=0;i<ncols;i++)
		 inputCol[i] = (gdf_column*) malloc(sizeof(gdf_column));
 
	 gdf_column *outputCol;
	 outputCol = (gdf_column*) malloc(sizeof(gdf_column));
 
	 inputCol[0]->dtype = GDF_INT8;
	 inputCol[0]->size = nrows;
 
	 inputCol[1]->dtype = GDF_INT16;
	 inputCol[1]->size = nrows;
 
	 inputCol[2]->dtype = GDF_INT32;
	 inputCol[2]->size = nrows;
 
	 inputCol[3]->dtype = GDF_INT64;
	 inputCol[3]->size = nrows;
 
	 inputCol[4]->dtype = GDF_FLOAT32;
	 inputCol[4]->size = nrows;
 
	 inputCol[5]->dtype = GDF_FLOAT64;
	 inputCol[5]->size = nrows;
 
	 outputCol->dtype = GDF_INT32;
	 outputCol->size = nrows;
 
	 // Input Data
	 std::vector<int8_t> inputData1(nrows);
	 inputData1[0] = 0;
	 inputData1[1] = 1;
	 inputData1[2] = 2;
	 inputData1[3] = 3;
	 inputData1[4] = 0;
 
	 std::vector<int16_t> inputData2(nrows);
	 inputData2[0] = 4;
	 inputData2[1] = 3;
	 inputData2[2] = 54;
	 inputData2[3] = 7;
	 inputData2[4] = 4;
 
	 std::vector<int32_t> inputData3(nrows);
	 inputData3[0] = 28;
	 inputData3[1] = 543453;
	 inputData3[2] = 357894;
	 inputData3[3] = 456864;
	 inputData3[4] = 28;
 
	 std::vector<int64_t> inputData4(nrows);
	 inputData4[0] = 28467476447;
	 inputData4[1] = 43348947784;
	 inputData4[2] = 76343746754;
	 inputData4[3] = 91125744743;
	 inputData4[4] = 28467476447;
 
	 std::vector<float> inputData5(nrows);
	 inputData5[0] = 3.0;
	 inputData5[1] = 11.0;
	 inputData5[2] = 24.0;
	 inputData5[3] = 83.0;
	 inputData5[4] = 3.0;
 
	 std::vector<double> inputData6(nrows);
	 inputData6[0] = 343.01;
	 inputData6[1] = 112.04;
	 inputData6[2] = 298.67;
	 inputData6[3] = 786.34;
	 inputData6[4] = 343.01;
 
	 Vector<gdf_valid_type> inputValidDev(1,0);
 
	 Vector<int8_t> intputDataDev1(inputData1);
	 Vector<int16_t> intputDataDev2(inputData2);
	 Vector<int32_t> intputDataDev3(inputData3);
	 Vector<int64_t> intputDataDev4(inputData4);
	 Vector<float> intputDataDev5(inputData5);
	 Vector<double> intputDataDev6(inputData6);
 
	 Vector<int32_t> outDataDev(nrows);
	 Vector<gdf_valid_type> outputValidDev(1,0);
 
	 inputCol[0]->data = thrust::raw_pointer_cast(intputDataDev1.data());
	 inputCol[0]->valid = thrust::raw_pointer_cast(inputValidDev.data());
 
	 inputCol[1]->data = thrust::raw_pointer_cast(intputDataDev2.data());
	 inputCol[1]->valid = thrust::raw_pointer_cast(inputValidDev.data());
 
	 inputCol[2]->data = thrust::raw_pointer_cast(intputDataDev3.data());
	 inputCol[2]->valid = thrust::raw_pointer_cast(inputValidDev.data());
 
	 inputCol[3]->data = thrust::raw_pointer_cast(intputDataDev4.data());
	 inputCol[3]->valid = thrust::raw_pointer_cast(inputValidDev.data());
 
	 inputCol[4]->data = thrust::raw_pointer_cast(intputDataDev5.data());
	 inputCol[4]->valid = thrust::raw_pointer_cast(inputValidDev.data());
 
	 inputCol[5]->data = thrust::raw_pointer_cast(intputDataDev6.data());
	 inputCol[5]->valid = thrust::raw_pointer_cast(inputValidDev.data());
 
	 outputCol->data = thrust::raw_pointer_cast(outDataDev.data());
	 outputCol->valid = thrust::raw_pointer_cast(outputValidDev.data());
 
	 {
		 gdf_hash_func hash = GDF_HASH_MURMUR3;
		 gdf_error gdfError = gdf_hash(ncols, inputCol, hash, outputCol);
 
		 EXPECT_TRUE( gdfError == GDF_SUCCESS );
		 EXPECT_FALSE( gdfError == GDF_CUDA_ERROR );
		 EXPECT_FALSE( gdfError == GDF_UNSUPPORTED_DTYPE );
		 EXPECT_FALSE( gdfError == GDF_COLUMN_SIZE_MISMATCH );
 
		 std::vector<int32_t> results(nrows);
		 thrust::copy(outDataDev.begin(), outDataDev.end(), results.begin());
 
		 EXPECT_TRUE( results[0] == results[nrows-1]);
	 }
 }
 