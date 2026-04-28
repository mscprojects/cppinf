#include <fmt/format.h>
#include <spdlog/spdlog.h>

int main() {
  spdlog::info("Starting cppinf");
  fmt::print("Hello, {}!\n", "cppinf");
  return 0;
}
