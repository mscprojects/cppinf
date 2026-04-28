#include <fmt/format.h>
#include <spdlog/spdlog.h>

#include "greeting.h"

int main() {
  spdlog::info("Starting cppinf");
  fmt::print("{}\n", BuildGreeting("cppinf"));
  return 0;
}
