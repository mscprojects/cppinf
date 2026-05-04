#include "common/checked.h"

#include <limits>
#include <stdexcept>

#include <fmt/format.h>

namespace cppinf::common {

std::size_t checked_non_negative_dim_to_size(std::int64_t dim, std::string_view field_name) {
    if (dim < 0) {
        throw std::invalid_argument(fmt::format("{} must be non-negative.", field_name));
    }

    return static_cast<std::size_t>(dim);
}

std::size_t checked_positive_dim_to_size(std::int64_t dim, std::string_view field_name) {
    const auto value = checked_non_negative_dim_to_size(dim, field_name);
    if (value == 0) {
        throw std::invalid_argument(fmt::format("{} must be non-zero.", field_name));
    }

    return value;
}

std::size_t checked_positive_size(std::size_t value, std::string_view field_name) {
    if (value == 0) {
        throw std::invalid_argument(fmt::format("{} must be non-zero.", field_name));
    }

    return value;
}

std::int64_t checked_size_to_dim(std::size_t value, std::string_view field_name) {
    if (value > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
        throw std::overflow_error(fmt::format("{} does not fit in int64_t.", field_name));
    }

    return static_cast<std::int64_t>(value);
}

std::size_t checked_u64_to_size(std::uint64_t value, std::string_view field_name) {
    if (value > std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error(fmt::format("{} does not fit in size_t.", field_name));
    }

    return static_cast<std::size_t>(value);
}

} // namespace cppinf::common
