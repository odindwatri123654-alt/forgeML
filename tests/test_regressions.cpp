// Регрессионные тесты: по одному на каждый дефект из qa/QA_REPORT.md.
#include "test_framework.h"

#include <forge/forge.h>

#include <cmath>
#include <cstdio>
#include <limits>
#include <sstream>
#include <thread>
#include <utility>

using namespace forge;

// D1: длинный граф не должен переполнять стек ни в backward, ни при удалении.
TEST(regression_d1_deep_graph_backward_and_destroy) {
    auto x = Tensor::full({}, 1.0f).requires_grad_();
    Tensor y = x;
    for (int i = 0; i < 200000; ++i) y = y + 1.0f;
    y.backward();
    CHECK(x.grad().item() == 1.0f);
    y = Tensor();  // удаление графа из 200 000 узлов
    CHECK(x.grad().item() == 1.0f);
}

// D2: ось размера 1 растягивается и до размера 0.
TEST(regression_d2_empty_broadcasting) {
    CHECK(broadcast_shapes({0}, {1}) == (Shape{0}));
    CHECK(broadcast_shapes({0, 3}, {}) == (Shape{0, 3}));
    auto e = Tensor::zeros({0, 3});
    CHECK((e * 2.0f).shape() == (Shape{0, 3}));
    CHECK((e + Tensor::ones({3})).shape() == (Shape{0, 3}));
    CHECK_THROWS(broadcast_shapes({0}, {2}));
}

// D3: сумма по оси на длинной оси так же точна, как полная сумма.
TEST(regression_d3_sum_dim_precision) {
    auto x = Tensor::full({1, 10000000}, 0.1f);
    CHECK_NEAR(sum(x, 1).item(), 1e6, 1.0);
    CHECK_NEAR(mean(x, 1).item(), 0.1, 1e-6);
}

// D4: неудачная загрузка не меняет модель.
TEST(regression_d4_failed_load_is_atomic) {
    nn::Sequential a;
    a.add<nn::Linear>(3, 4);
    a.add<nn::Linear>(4, 2);
    a.save("forge_regression.bin");
    nn::Sequential b;
    b.add<nn::Linear>(3, 4);
    b.add<nn::Linear>(4, 5);
    const Tensor first = b.parameters()[0];
    float before = first.at({0, 0});
    CHECK_THROWS(b.load("forge_regression.bin"));
    CHECK(first.at({0, 0}) == before);
    std::remove("forge_regression.bin");
}

// D5: генератор случайных чисел безопасен из нескольких потоков
// (полная проверка — ThreadSanitizer; здесь — что ничего не ломается).
TEST(regression_d5_parallel_random) {
    std::vector<std::thread> threads;
    std::vector<float> sums(4);
    for (std::size_t t = 0; t < 4; ++t) {
        threads.emplace_back([t, &sums] {
            nn::Linear layer(64, 64);
            sums[t] = sum(Tensor::rand({10000})).item();
        });
    }
    for (auto& th : threads) th.join();
    for (float s : sums) CHECK(s > 4500.0f && s < 5500.0f);
}

// D6: NaN проходит через max и argmax, как в PyTorch.
TEST(regression_d6_nan_in_max) {
    float nan = std::numeric_limits<float>::quiet_NaN();
    auto x = Tensor::from_vector({1, nan, 3, 4, 5, 6}, {2, 3});
    auto m = max(x, 1);
    CHECK(std::isnan(m.at({0})));
    CHECK(m.at({1}) == 6.0f);
    CHECK(argmax(x, 1).at({0}) == 1.0f);
}

// D7: изменение входа между forward и backward — понятная ошибка.
TEST(regression_d7_modified_input_detected) {
    auto w = Tensor::from_vector({1}, {1}).requires_grad_();
    optim::SGD opt({w}, 0.1f);
    auto loss = sum(w * w);
    w.zero_grad();
    sum(w * 3.0f).backward();  // какой-то другой граф
    opt.step();                // меняет w до backward первого графа
    CHECK_THROWS(loss.backward());
}

// D7, обратная сторона: обычное чтение не считается изменением.
TEST(regression_d7_reading_is_not_modification) {
    auto w = Tensor::from_vector({1, 2}, {2}).requires_grad_();
    auto loss = sum(w * w);
    const Tensor& view = w;
    float a = view.at({0});        // const at()
    float b = loss.item();         // item()
    std::ostringstream os;
    os << w;                       // печать
    (void)a;
    (void)b;
    loss.backward();               // не должно бросать
    CHECK(w.grad().at({1}) == 4.0f);
}

// D8: дробные и NaN-метки отклоняются.
TEST(regression_d8_fractional_labels) {
    auto logits = Tensor::zeros({2, 3});
    CHECK_THROWS(nn::cross_entropy(logits, Tensor::from_vector({0, 1.5f}, {2})));
    CHECK_THROWS(nn::cross_entropy(
        logits, Tensor::from_vector({0, std::numeric_limits<float>::quiet_NaN()}, {2})));
    CHECK(std::isfinite(nn::cross_entropy(logits, Tensor::from_vector({0, 2}, {2})).item()));
}

// D9: понятное сообщение для Linear с нулевым размером.
TEST(regression_d9_linear_zero_features) {
    try {
        nn::Linear bad(0, 5);
        CHECK(false);
    } catch (const std::invalid_argument& e) {
        CHECK(std::string(e.what()).find("Linear") != std::string::npos);
    }
    CHECK_THROWS(nn::Linear(5, 0));
}
