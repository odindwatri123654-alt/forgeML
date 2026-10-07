#pragma once
// Минимальный фреймворк для тестов (вместо GoogleTest — без зависимостей).
//
//   TEST(my_test) { CHECK(1 + 1 == 2); }
//
// Каждый TEST регистрирует себя в общем списке ещё до main(),
// а test_main.cpp запускает их все по очереди.

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace testing {

struct TestCase {
    const char* name;
    void (*fn)();
};

// Список всех тестов. static внутри функции — создаётся при первом обращении,
// поэтому не важно, в каком порядке инициализируются тестовые файлы.
inline std::vector<TestCase>& registry() {
    static std::vector<TestCase> tests;
    return tests;
}

// Конструктор этого объекта добавляет тест в список.
struct Registrar {
    Registrar(const char* name, void (*fn)()) { registry().push_back({name, fn}); }
};

// Провал проверки — исключение со строкой "файл:строка: что не так".
struct Failure : std::runtime_error {
    using std::runtime_error::runtime_error;
};

inline std::string where(const char* file, int line) {
    return std::string(file) + ":" + std::to_string(line) + ": ";
}

} // namespace testing

#define TEST(name)                                                    \
    static void name();                                               \
    static const testing::Registrar name##_registrar(#name, &name);   \
    static void name()

#define CHECK(cond)                                                                   \
    do {                                                                              \
        if (!(cond)) throw testing::Failure(testing::where(__FILE__, __LINE__) +      \
                                            "CHECK(" #cond ") failed");               \
    } while (0)

#define CHECK_NEAR(a, b, tol)                                                         \
    do {                                                                              \
        double va_ = (a), vb_ = (b);                                                  \
        if (std::fabs(va_ - vb_) > (tol))                                             \
            throw testing::Failure(testing::where(__FILE__, __LINE__) + #a " = " +    \
                                   std::to_string(va_) + ", " #b " = " +              \
                                   std::to_string(vb_));                              \
    } while (0)

#define CHECK_THROWS(expr)                                                            \
    do {                                                                              \
        bool thrown_ = false;                                                         \
        try {                                                                         \
            (void)(expr);                                                             \
        } catch (const std::exception&) {                                             \
            thrown_ = true;                                                           \
        }                                                                             \
        if (!thrown_) throw testing::Failure(testing::where(__FILE__, __LINE__) +     \
                                             "expected exception from " #expr);       \
    } while (0)
