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

#include <cudf/scalar/scalar.hpp>

#include <rmm/device_buffer.hpp>

#include <string>

namespace cudf {

// Copy constructor
scalar::scalar(scalar const &other)
    : _type{other._type},
      _is_valid{other._is_valid} {}

// Move constructor
scalar::scalar(scalar &&other)
    : _type{other._type},
      _is_valid{std::move(other._is_valid)}
{
  other._type = data_type{EMPTY};
}

string_scalar::string_scalar(std::string const& string, bool is_valid)
 : scalar(data_type(STRING), is_valid), _data(string.data(), string.size())
{}

std::string string_scalar::value() const {
  std::string result;
  result.resize(_data.size());
  CUDA_TRY(cudaMemcpy(&result[0], _data.data(), _data.size(), cudaMemcpyDeviceToHost));
  return result;
}

}  // namespace cudf