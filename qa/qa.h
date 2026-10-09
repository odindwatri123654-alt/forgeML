#pragma once
// QA-набор ForgeML: тесты "чёрного ящика", написанные по README.
// Каждый тест запускается в отдельном процессе, поэтому падение программы
// (а не исключение) фиксируется как CRASH и не мешает остальным тестам.

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace qa {

struct Case {
    const char* name;
    const char* group;
    const char* description;
    void (*fn)();
};

inline std::vector<Case>& cases() {
    static std::vector<Case> all;
    return all;
}

struct Registrar {
    Registrar(const char* name, const char* group, const char* description, void (*fn)()) {
        cases().push_back({name, group, description, fn});
    }
};

struct Failure : std::runtime_error {
    using std::runtime_error::runtime_error;
};

template <typename T>
std::string str(const T& v) {
    std::ostringstream os;
    os << v;
    return os.str();
}

} // namespace qa

#define QA_TEST(name, group, description)                                         \
    static void name();                                                           \
    static const qa::Registrar name##_reg(#name, group, description, &name);      \
    static void name()

#define EXPECT(cond, msg)                                                         \
    do {                                                                          \
        if (!(cond)) throw qa::Failure(std::string("ожидалось: ") + (msg) +      \
                                       "  [" #cond "]");                          \
    } while (0)

#define EXPECT_NEAR(a, b, tol, msg)                                               \
    do {                                                                          \
        double qa_a = (a), qa_b = (b);                                            \
        if (!(std::fabs(qa_a - qa_b) <= (tol)))                                   \
            throw qa::Failure(std::string("ожидалось: ") + (msg) + "  (получено " + \
                              qa::str(qa_a) + ", нужно " + qa::str(qa_b) + ")");   \
    } while (0)

// Ожидаем понятное исключение (std::exception), а не падение программы.
#define EXPECT_THROWS(expr, msg)                                                  \
    do {                                                                          \
        bool qa_thrown = false;                                                   \
        try {                                                                     \
            (void)(expr);                                                         \
        } catch (const std::exception&) {                                         \
            qa_thrown = true;                                                     \
        }                                                                         \
        if (!qa_thrown) throw qa::Failure(std::string("ожидалось исключение: ") + (msg)); \
    } while (0)
