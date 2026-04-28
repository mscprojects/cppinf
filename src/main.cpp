#include <cstdio>
#include <string_view>
#include <vector>

#include <fmt/format.h>

#include "cli/cli_app.h"

int main(int argc, char* argv[]) {
    std::vector<std::string_view> args;
    args.reserve(argc > 0 ? static_cast<std::size_t>(argc - 1) : 0);
    for (int index = 1; index < argc; ++index) {
        args.push_back(argv[index]);
    }

    const cppinf::cli::CliResult result = cppinf::cli::run(args);
    if (result.exit_code == 0) {
        fmt::print("{}", result.output);
    } else {
        fmt::print(stderr, "{}", result.output);
    }

    return result.exit_code;
}
