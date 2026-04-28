#pragma once

#include <cstdint>

namespace cppinf::tensors {

std::uint16_t float_to_bfloat16_bits(float value);
float bfloat16_bits_to_float(std::uint16_t bits);

} // namespace cppinf::tensors
