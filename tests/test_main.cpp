#include "test_framework.h"

#include <exception>
#include <iostream>

int main() {
    int failed = 0;
    for (const testing::TestCase& test : testing::registry()) {
        try {
            test.fn();
            std::cout << "[  OK  ] " << test.name << "\n";
        } catch (const std::exception& e) {
            ++failed;
            std::cout << "[ FAIL ] " << test.name << "\n         " << e.what() << "\n";
        }
    }
    std::size_t total = testing::registry().size();
    std::cout << "\n" << (total - static_cast<std::size_t>(failed)) << "/" << total
              << " tests passed\n";
    return failed == 0 ? 0 : 1;
}
