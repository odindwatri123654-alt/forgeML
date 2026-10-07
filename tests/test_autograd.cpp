#include "test_framework.h"

#include <forge/autograd.h>
#include <forge/ops.h>

#include <algorithm>
#include <functional>

using namespace forge;

namespace {

// Проверка градиента: аналитический (backward) против численного
//   df/dx ≈ (f(x + eps) - f(x - eps)) / (2 * eps)
// f возвращает скаляр. Чтобы проверить не только сумму, выход операции
// умножается на случайные веса w: f = sum(op(inputs) * w).
void gradcheck(const std::function<Tensor(const std::vector<Tensor>&)>& op,
               std::vector<Tensor> inputs, float eps = 1e-2f, float tol = 2e-2f) {
    for (Tensor& x : inputs) x.requires_grad_(true);
    Tensor w;
    {
        NoGradGuard no_grad;
        w = Tensor::randn(op(inputs).shape());
    }
    auto f = [&]() { return sum(op(inputs) * w); };

    for (Tensor& x : inputs) x.zero_grad();
    f().backward();

    for (std::size_t n = 0; n < inputs.size(); ++n) {
        Tensor& x = inputs[n];
        Tensor analytic = x.grad();
        CHECK(analytic.defined());
        CHECK(analytic.shape() == x.shape());
        NoGradGuard no_grad;
        for (std::size_t i = 0; i < x.numel(); ++i) {
            float original = x.data()[i];
            x.data()[i] = original + eps;
            float plus = f().item();
            x.data()[i] = original - eps;
            float minus = f().item();
            x.data()[i] = original;
            float numeric = (plus - minus) / (2.0f * eps);
            float a = analytic.contiguous().data()[i];
            float scale = std::max(1.0f, std::fabs(numeric));
            if (std::fabs(a - numeric) > tol * scale) {
                throw testing::Failure("gradcheck: input " + std::to_string(n) + ", element " +
                                       std::to_string(i) + ": analytic " + std::to_string(a) +
                                       " vs numeric " + std::to_string(numeric));
            }
        }
    }
}

Tensor positive(const Shape& shape) { return Tensor::rand(shape) + 0.5f; }

} // namespace

TEST(autograd_simple_chain) {
    // y = a*b + b, dy/da = b = 3, dy/db = a + 1 = 3
    auto a = Tensor::full({}, 2.0f).requires_grad_();
    auto b = Tensor::full({}, 3.0f).requires_grad_();
    auto y = a * b + b;
    y.backward();
    CHECK(a.grad().item() == 3.0f);
    CHECK(b.grad().item() == 3.0f);
    CHECK(a.is_leaf() && !y.is_leaf());
}

TEST(autograd_tensor_used_twice) {
    auto x = Tensor::full({}, 3.0f).requires_grad_();
    auto y = x * x;  // dy/dx = 2x = 6
    y.backward();
    CHECK(x.grad().item() == 6.0f);
}

TEST(autograd_gradients_accumulate) {
    auto x = Tensor::full({}, 1.0f).requires_grad_();
    (x * 2.0f).backward();
    (x * 2.0f).backward();
    CHECK(x.grad().item() == 4.0f);
    x.zero_grad();
    CHECK(!x.grad().defined());
}

TEST(autograd_no_grad_and_detach) {
    auto x = Tensor::ones({2}).requires_grad_();
    {
        NoGradGuard guard;
        CHECK(!(x * 2.0f).requires_grad());
    }
    CHECK((x * 2.0f).requires_grad());
    CHECK(!x.detach().requires_grad());
    CHECK_THROWS((x * 2.0f).requires_grad_());            // не лист
    CHECK_THROWS(Tensor::ones({2}).backward());           // не требует градиента
    CHECK_THROWS((x * 2.0f).backward());                  // не скаляр
}

TEST(autograd_broadcast_backward) {
    // bias [3] прибавлен к [2,3]: градиент bias — сумма по строкам
    auto x = Tensor::ones({2, 3});
    auto b = Tensor::zeros({3}).requires_grad_();
    sum(x + b).backward();
    CHECK(b.grad().shape() == (Shape{3}));
    CHECK(b.grad().at({0}) == 2.0f);
}

TEST(gradcheck_elementwise) {
    manual_seed(1);
    gradcheck([](auto& in) { return in[0] + in[1]; }, {Tensor::randn({2, 3}), Tensor::randn({2, 3})});
    gradcheck([](auto& in) { return in[0] - in[1]; }, {Tensor::randn({2, 3}), Tensor::randn({3})});
    gradcheck([](auto& in) { return in[0] * in[1]; }, {Tensor::randn({2, 3}), Tensor::randn({2, 1})});
    gradcheck([](auto& in) { return in[0] / in[1]; }, {Tensor::randn({2, 3}), positive({3})});
    gradcheck([](auto& in) { return -in[0]; }, {Tensor::randn({4})});
}

TEST(gradcheck_unary) {
    manual_seed(2);
    gradcheck([](auto& in) { return exp(in[0]); }, {Tensor::randn({2, 3})});
    gradcheck([](auto& in) { return log(in[0]); }, {positive({2, 3})});
    gradcheck([](auto& in) { return sigmoid(in[0]); }, {Tensor::randn({2, 3})});
    gradcheck([](auto& in) { return tanh(in[0]); }, {Tensor::randn({2, 3})});
    gradcheck([](auto& in) { return pow(in[0], 3.0f); }, {Tensor::randn({2, 3})});
    gradcheck([](auto& in) { return pow(in[0], 0.5f); }, {positive({2, 3})});
    // relu: значения подальше от излома в нуле
    gradcheck([](auto& in) { return relu(in[0]); },
              {Tensor::from_vector({-2, -0.5f, 0.7f, 1.5f, -1, 3}, {2, 3})});
}

TEST(gradcheck_reductions) {
    manual_seed(3);
    gradcheck([](auto& in) { return sum(in[0]); }, {Tensor::randn({2, 3})});
    gradcheck([](auto& in) { return sum(in[0], 0); }, {Tensor::randn({2, 3})});
    gradcheck([](auto& in) { return sum(in[0], 1, true); }, {Tensor::randn({2, 3, 2})});
    gradcheck([](auto& in) { return mean(in[0]); }, {Tensor::randn({2, 3})});
    gradcheck([](auto& in) { return mean(in[0], 1); }, {Tensor::randn({2, 3})});
    gradcheck([](auto& in) { return max(in[0], 1); }, {Tensor::randn({3, 4})});
}

TEST(gradcheck_views) {
    manual_seed(4);
    gradcheck([](auto& in) { return in[0].transpose(0, 1); }, {Tensor::randn({2, 3})});
    gradcheck([](auto& in) { return in[0].reshape({3, 2}); }, {Tensor::randn({2, 3})});
    gradcheck([](auto& in) { return in[0].permute({2, 0, 1}); }, {Tensor::randn({2, 3, 4})});
    gradcheck([](auto& in) { return in[0].expand({4, 3}); }, {Tensor::randn({1, 3})});
    gradcheck([](auto& in) { return in[0].transpose(0, 1).reshape({6}); }, {Tensor::randn({2, 3})});
    gradcheck([](auto& in) { return in[0].unsqueeze(0); }, {Tensor::randn({3})});
}

TEST(gradcheck_matmul_and_softmax) {
    manual_seed(5);
    gradcheck([](auto& in) { return matmul(in[0], in[1]); },
              {Tensor::randn({3, 4}), Tensor::randn({4, 2})});
    gradcheck([](auto& in) { return matmul(in[0], in[1]) + in[2]; },
              {Tensor::randn({3, 4}), Tensor::randn({4, 2}), Tensor::randn({2})});
    gradcheck([](auto& in) { return softmax(in[0], 1); }, {Tensor::randn({2, 4})});
    gradcheck([](auto& in) { return log_softmax(in[0], 1); }, {Tensor::randn({2, 4})});
}
