#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace cppinf::tensors {

class Shape {
  public:
    Shape() = default;
    explicit Shape(std::vector<std::int64_t> dims);

    const std::vector<std::int64_t>& dims() const;
    std::size_t rank() const;
    std::size_t num_elements() const;

    bool operator==(const Shape&) const = default;

  private:
    std::vector<std::int64_t> dims_;
};

std::string to_string(const Shape& shape);

} // namespace cppinf::tensors
