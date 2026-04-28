#include "bfloat16.h"

#include <cstdint>
#include <cstring>

namespace cppinf::tensors {

std::uint16_t float_to_bfloat16_bits(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));

    const std::uint32_t least_significant_bit = (bits >> 16U) & 1U;
    bits += 0x7fffU + least_significant_bit;
    return static_cast<std::uint16_t>(bits >> 16U);
}

float bfloat16_bits_to_float(std::uint16_t bits) {
    const std::uint32_t float_bits = static_cast<std::uint32_t>(bits) << 16U;
    float value = 0.0f;
    std::memcpy(&value, &float_bits, sizeof(value));
    return value;
}

} // namespace cppinf::tensors
