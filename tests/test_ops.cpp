#include "test_framework.h"

#include <forge/ops.h>

using namespace forge;

namespace {
bool same(const Tensor& t, const std::vector<float>& expected, float tol = 1e-5f) {
    Tensor c = t.contiguous();
    if (c.numel() != expected.size()) return false;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (std::fabs(c.data()[i] - expected[i]) > tol) return false;
    }
    return true;
}
} // namespace

TEST(ops_broadcast_shapes) {
    CHECK(broadcast_shapes({2, 3}, {3}) == (Shape{2, 3}));
    CHECK(broadcast_shapes({2, 1}, {1, 3}) == (Shape{2, 3}));
    CHECK(broadcast_shapes({}, {4}) == (Shape{4}));
    CHECK_THROWS(broadcast_shapes({2, 3}, {2}));
}

TEST(ops_elementwise_with_broadcasting) {
    auto x = Tensor::from_vector({1, 2, 3, 4, 5, 6}, {2, 3});
    auto b = Tensor::from_vector({10, 20, 30}, {3});
    CHECK(same(x + b, {11, 22, 33, 14, 25, 36}));
    CHECK(same(x - 1.0f, {0, 1, 2, 3, 4, 5}));
    CHECK(same(2.0f * x, {2, 4, 6, 8, 10, 12}));
    CHECK(same(x / x, {1, 1, 1, 1, 1, 1}));
    CHECK(same(1.0f - x, {0, -1, -2, -3, -4, -5}));
    CHECK(same(-x, {-1, -2, -3, -4, -5, -6}));
}

TEST(ops_unary) {
    auto x = Tensor::from_vector({-1, 0, 2}, {3});
    CHECK(same(relu(x), {0, 0, 2}));
    CHECK(same(pow(x, 2), {1, 0, 4}));
    CHECK(same(log(exp(x)), {-1, 0, 2}));
    CHECK(same(sigmoid(Tensor::zeros({1})), {0.5f}));
    CHECK(same(tanh(Tensor::zeros({1})), {0.0f}));
}

TEST(ops_reductions) {
    auto x = Tensor::from_vector({1, 2, 3, 4, 5, 6}, {2, 3});
    CHECK(sum(x).item() == 21.0f);
    CHECK(same(sum(x, 0), {5, 7, 9}));
    CHECK(same(sum(x, 1), {6, 15}));
    CHECK(sum(x, 1, true).shape() == (Shape{2, 1}));
    CHECK(mean(x).item() == 3.5f);
    CHECK(same(mean(x, 1), {2, 5}));
    CHECK(same(max(x, 0), {4, 5, 6}));
    CHECK(same(argmax(Tensor::from_vector({1, 9, 3, 7, 2, 0}, {2, 3}), 1), {1, 0}));
    CHECK_THROWS(sum(x, 2));
    CHECK_THROWS(mean(x, 5));
}

TEST(ops_reduction_on_transposed) {
    auto x = Tensor::from_vector({1, 2, 3, 4, 5, 6}, {2, 3}).transpose(0, 1);
    CHECK(same(sum(x, 1), {5, 7, 9}));
}

TEST(ops_softmax) {
    auto x = Tensor::from_vector({1, 2, 3, 1000, 1000, 1000}, {2, 3});
    auto s = softmax(x, 1);
    CHECK(same(sum(s, 1), {1, 1}));
    CHECK(same(s, {0.0900306f, 0.244728f, 0.665241f, 1.0f / 3, 1.0f / 3, 1.0f / 3}));
    CHECK(same(exp(log_softmax(x, 1)), {0.0900306f, 0.244728f, 0.665241f, 1.0f / 3, 1.0f / 3,
                                        1.0f / 3}));
}

TEST(ops_matmul) {
    auto a = Tensor::from_vector({1, 2, 3, 4, 5, 6}, {2, 3});
    auto b = Tensor::from_vector({7, 8, 9, 10, 11, 12}, {3, 2});
    CHECK(same(matmul(a, b), {58, 64, 139, 154}));
    CHECK(same(matmul(a, a.transpose(0, 1)), {14, 32, 32, 77}));
    CHECK(same(matmul(Tensor::eye(2), Tensor::from_vector({5, 6, 7, 8}, {2, 2})), {5, 6, 7, 8}));
    CHECK_THROWS(matmul(a, a));
    CHECK_THROWS(matmul(Tensor::ones({3}), a));
}

TEST(ops_large_matmul_matches_naive) {
    manual_seed(3);
    auto a = Tensor::randn({70, 50});
    auto b = Tensor::randn({50, 40});
    auto c = matmul(a, b);
    for (std::size_t i = 0; i < 70; i += 13) {
        for (std::size_t j = 0; j < 40; j += 7) {
            double ref = 0;
            for (std::size_t k = 0; k < 50; ++k) ref += double(a.at({i, k})) * b.at({k, j});
            CHECK_NEAR(c.at({i, j}), ref, 1e-3);
        }
    }
}
