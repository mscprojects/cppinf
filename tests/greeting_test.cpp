#include <exception>
#include <iostream>
#include <string>

#include "greeting.h"
#include "test.h"

int main() {
  try {
    ExpectEqual(std::string("Hello, cppinf!"),
                BuildGreeting("cppinf"),
                "BuildGreeting should format the requested name.");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "cppinf_tests failed: " << error.what() << '\n';
    return 1;
  }
}
