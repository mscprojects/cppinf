#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace cppinf::common {

// Converts a non-negative tensor dimension to size_t, or throws if the dimension is negative.
std::size_t checked_non_negative_dim_to_size(std::int64_t dim, std::string_view field_name);

// Converts a positive tensor dimension to size_t, or throws if the dimension is zero or negative.
std::size_t checked_positive_dim_to_size(std::int64_t dim, std::string_view field_name);

// Returns a positive size_t value, or throws if the value is zero.
std::size_t checked_positive_size(std::size_t value, std::string_view field_name);

// Converts a size_t extent to int64_t shape storage, or throws if the value would overflow.
std::int64_t checked_size_to_dim(std::size_t value, std::string_view field_name);

// Converts a uint64_t file field to size_t, or throws if the value would overflow.
std::size_t checked_u64_to_size(std::uint64_t value, std::string_view field_name);

} // namespace cppinf::common
