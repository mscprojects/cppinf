#include "shape.h"

#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace cppinf::tensors {

Shape::Shape(std::vector<std::int64_t> dims) : dims_(std::move(dims)) {
    for (const std::int64_t dim : dims_) {
        if (dim < 0) {
            throw std::invalid_argument("Shape dimensions must be non-negative.");
        }
    }
}

const std::vector<std::int64_t>& Shape::dims() const {
    return dims_;
}

std::size_t Shape::rank() const {
    return dims_.size();
}

std::size_t Shape::num_elements() const {
    std::size_t num_elements = 1;
    for (const std::int64_t dim : dims_) {
        const std::size_t dim_size = static_cast<std::size_t>(dim);
        if (dim_size != 0 && num_elements > std::numeric_limits<std::size_t>::max() / dim_size) {
            throw std::overflow_error("Shape element count overflowed.");
        }

        num_elements *= dim_size;
    }

    return num_elements;
}

std::string to_string(const Shape& shape) {
    std::ostringstream stream;
    stream << '[';

    for (std::size_t index = 0; index < shape.dims().size(); ++index) {
        if (index != 0) {
            stream << ", ";
        }

        stream << shape.dims()[index];
    }

    stream << ']';
    return stream.str();
}

} // namespace cppinf::tensors
