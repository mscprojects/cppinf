#include <cstdio>
#include <string_view>
#include <vector>

#include <fmt/format.h>

#include "cli/cli_app.h"

int main(int argc, char* argv[]) {
    using cppinf::cli::run_with_output_writer;

    std::vector<std::string_view> args;
    args.reserve(argc > 0 ? static_cast<std::size_t>(argc - 1) : 0);
    for (int index = 1; index < argc; ++index) {
        args.emplace_back(argv[index]);
    }

    const auto result = run_with_output_writer(args, [](std::string_view chunk) {
        fmt::print("{}", chunk);
        std::fflush(stdout);
    });
    if (result.exit_code == 0) {
        fmt::print("{}", result.output);
        std::fflush(stdout);
    } else {
        fmt::print(stderr, "{}", result.output);
    }

    return result.exit_code;
}
