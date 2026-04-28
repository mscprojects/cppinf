#include "greeting.h"

#include <fmt/format.h>

std::string BuildGreeting(std::string_view name) {
  return fmt::format("Hello, {}!", name);
}
