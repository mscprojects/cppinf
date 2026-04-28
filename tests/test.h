#pragma once

#include <stdexcept>
#include <string>
#include <string_view>

inline void Expect(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error(std::string(message));
  }
}

template <typename Expected, typename Actual>
inline void ExpectEqual(const Expected& expected,
                        const Actual& actual,
                        std::string_view message) {
  Expect(expected == actual, message);
}
