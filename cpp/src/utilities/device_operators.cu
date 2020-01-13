/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <cudf/detail/utilities/device_operators.cuh>

namespace cudf {

__constant__ char max_string_sentinel[5]{"\xF7\xBF\xBF\xBF"};

char const* get_max_string_sentinel() {
  const char* psentinel{nullptr};
  cudaGetSymbolAddress((void**)&psentinel, max_string_sentinel);
  return psentinel;
}

} // namespace cudf

