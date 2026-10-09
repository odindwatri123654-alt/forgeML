#include "qa.h"

#include <forge/forge.h>

#include <functional>

using namespace forge;

namespace {

std::vector<float> values(const Tensor& t) {
    Tensor c = t.contiguous();
    return std::vector<float>(c.data(), c.data() + c.numel());
}

// Своя численная производная (тестировщик не доверяет внутренним инструментам).
std::vector<float> numeric_grad(const std::function<float()>& f, Tensor& x, float eps = 1e-2f) {
    NoGradGuard ng;
    std::vector<float> g(x.numel());
    for (std::size_t i = 0; i < x.numel(); ++i) {
        float old = x.data()[i];
        x.data()[i] = old + eps;
        float p = f();
        x.data()[i] = old - eps;
        float m = f();
        x.data()[i] = old;
        g[i] = (p - m) / (2 * eps);
    }
    return g;
}

void expect_close(const std::vector<float>& a, const std::vector<float>& b, float tol,
                  const std::string& what) {
    EXPECT(a.size() == b.size(), what + ": размеры совпадают");
    for (std::size_t i = 0; i < a.size(); ++i)
        EXPECT_NEAR(a[i], b[i], tol * std::max(1.0f, std::fabs(b[i])), what + " [" + qa::str(i) + "]");
}

} // namespace

// ------------------------------------------------------------------
// Autograd
// ------------------------------------------------------------------

QA_TEST(known_derivatives, "6. Autograd", "производные из учебника: x², mean, x·W, sin-подобные") {
    auto x = Tensor::from_vector({1, -2, 3}, {3}).requires_grad_();
    sum(x * x).backward();
    EXPECT(values(x.grad()) == (std::vector<float>{2, -4, 6}), "d/dx Σx² = 2x");

    auto y = Tensor::ones({4}).requires_grad_();
    mean(y).backward();
    EXPECT(values(y.grad()) == (std::vector<float>{0.25f, 0.25f, 0.25f, 0.25f}), "d/dx mean = 1/n");

    auto xin = Tensor::from_vector({1, 2, 3, 4}, {2, 2});
    auto W = Tensor::zeros({2, 3}).requires_grad_();
    sum(matmul(xin, W)).backward();
    // d/dW Σ(xW) = xᵀ·1 : строка i = сумма столбца i у x
    EXPECT(values(W.grad()) == (std::vector<float>{4, 4, 4, 6, 6, 6}), "d/dW Σ(x·W) = xᵀ·1");
}

QA_TEST(cross_entropy_gradient_formula, "6. Autograd",
        "градиент cross_entropy по логитам = (softmax − one_hot) / N") {
    manual_seed(3);
    auto logits = Tensor::randn({4, 5}).requires_grad_();
    auto labels = Tensor::from_vector({0, 4, 2, 2}, {4});
    nn::cross_entropy(logits, labels).backward();
    auto expected = values(softmax(logits.detach(), 1));
    std::vector<float> lab = values(labels);
    for (std::size_t i = 0; i < 4; ++i) expected[i * 5 + std::size_t(lab[i])] -= 1.0f;
    for (float& v : expected) v /= 4.0f;
    expect_close(values(logits.grad()), expected, 1e-5f, "dCE/dlogits");
}

QA_TEST(composite_network_numeric, "6. Autograd",
        "маленькая сеть tanh(xW+b)·sigmoid: backward совпадает с численной производной") {
    manual_seed(11);
    auto x = Tensor::randn({3, 4});
    auto W = Tensor::randn({4, 2}).requires_grad_();
    auto b = Tensor::randn({2}).requires_grad_();
    auto f = [&]() {
        auto h = matmul(x, W) + b;
        return sum(tanh(h) * sigmoid(h * 0.5f) + pow(h, 2) * 0.1f);
    };
    f().backward();
    expect_close(values(W.grad()), numeric_grad([&] { return f().item(); }, W), 2e-2f, "dW");
    expect_close(values(b.grad()), numeric_grad([&] { return f().item(); }, b), 2e-2f, "db");
}

QA_TEST(gradient_through_view_chain, "6. Autograd",
        "градиент проходит через цепочку transpose → reshape → expand → permute") {
    manual_seed(12);
    auto x = Tensor::randn({2, 3}).requires_grad_();
    auto w = Tensor::randn({4, 6});
    auto f = [&]() {
        auto v = x.transpose(0, 1).reshape({1, 6}).expand({4, 6});
        return sum(v * w);
    };
    f().backward();
    expect_close(values(x.grad()), numeric_grad([&] { return f().item(); }, x), 1e-2f, "dx через views");
}

QA_TEST(shared_layer_used_twice, "6. Autograd",
        "один и тот же слой применён дважды: градиенты складываются") {
    manual_seed(13);
    nn::Linear layer(3, 3);
    auto x = Tensor::randn({2, 3});
    auto f = [&]() { return sum(layer(layer(x))); };
    f().backward();
    auto analytic = values(layer.weight.grad());
    expect_close(analytic, numeric_grad([&] { return f().item(); }, layer.weight), 2e-2f, "dW при двух применениях");
}

QA_TEST(backward_twice_accumulates, "6. Autograd",
        "два backward одного графа подряд: по README градиенты копятся (×2)") {
    auto x = Tensor::full({}, 3.0f).requires_grad_();
    auto y = x * x;
    y.backward();
    y.backward();
    EXPECT(x.grad().item() == 12.0f, "2 * (2x) = 12");
}

QA_TEST(intermediate_has_no_grad, "6. Autograd", "у промежуточного тензора .grad() пустой (как PyTorch)") {
    auto x = Tensor::ones({2}).requires_grad_();
    auto h = x * 2.0f;
    sum(h * h).backward();
    EXPECT(!h.grad().defined(), "grad промежуточного не сохраняется");
    EXPECT(x.grad().defined(), "grad листа есть");
}

QA_TEST(requires_grad_off_and_detach, "6. Autograd",
        "requires_grad_(false) и detach() отключают градиент") {
    auto a = Tensor::ones({2}).requires_grad_();
    auto b = Tensor::ones({2}).requires_grad_();
    b.requires_grad_(false);
    sum(a * b).backward();
    EXPECT(a.grad().defined() && !b.grad().defined(), "градиент только у a");
    auto c = Tensor::ones({2}).requires_grad_();
    sum(c.detach() * a).backward();
    EXPECT(!c.grad().defined(), "через detach градиент не идёт");
}

QA_TEST(no_grad_guard_nested_and_exceptions, "6. Autograd",
        "NoGradGuard: вложенные блоки и исключение внутри блока корректно восстанавливают режим") {
    auto x = Tensor::ones({1}).requires_grad_();
    {
        NoGradGuard g1;
        {
            NoGradGuard g2;
        }
        EXPECT(!(x * 2.0f).requires_grad(), "после выхода из вложенного блока граф всё ещё выключен");
    }
    EXPECT((x * 2.0f).requires_grad(), "после выхода — включён");
    try {
        NoGradGuard g;
        throw std::runtime_error("boom");
    } catch (const std::exception&) {
    }
    EXPECT((x * 2.0f).requires_grad(), "после исключения граф снова включён");
}

QA_TEST(backward_with_explicit_grad, "6. Autograd", "backward(grad) для нескалярного выхода") {
    auto x = Tensor::from_vector({1, 2, 3}, {3}).requires_grad_();
    auto y = x * x;
    y.backward(Tensor::from_vector({1, 0, 10}, {3}));
    EXPECT(values(x.grad()) == (std::vector<float>{2, 0, 60}), "grad = 2x · v");
}

QA_TEST(max_ties_documented, "6. Autograd",
        "градиент max при равных элементах — во все равные (как написано в README)") {
    auto x = Tensor::from_vector({5, 5, 1}, {1, 3}).requires_grad_();
    sum(max(x, 1)).backward();
    EXPECT(values(x.grad()) == (std::vector<float>{1, 1, 0}), "[1, 1, 0]");
}

QA_TEST(modify_after_forward, "6. Autograd",
        "веса изменены между forward и backward: PyTorch выдаёт ошибку, а здесь?") {
    auto w = Tensor::from_vector({1}, {1}).requires_grad_();
    auto loss = sum(w * w);  // d/dw = 2w = 2 в момент forward
    w.data()[0] = 10.0f;     // "случайно" поменяли вес до backward
    loss.backward();
    EXPECT(w.grad().item() == 2.0f,
           "градиент для значения из forward (2) или ошибка; получено " + qa::str(w.grad().item()));
}

// ------------------------------------------------------------------
// Неправильное использование: должны быть понятные исключения
// ------------------------------------------------------------------

QA_TEST(undefined_tensor_everywhere, "7. Ошибки пользователя",
        "пустой Tensor() в операциях: исключение, а не падение") {
    Tensor t;
    EXPECT_THROWS(t + t, "t + t");
    EXPECT_THROWS(matmul(t, t), "matmul(t, t)");
    EXPECT_THROWS(t.item(), "t.item()");
    EXPECT_THROWS(t.backward(), "t.backward()");
    EXPECT_THROWS(relu(t), "relu(t)");
    EXPECT_THROWS(Tensor::ones({2}).grad().item(), "grad() до backward");
}

QA_TEST(index_errors, "7. Ошибки пользователя", "at() с неправильным числом индексов или за границей") {
    auto t = Tensor::zeros({2, 3});
    EXPECT_THROWS(t.at({2, 0}), "строка 2 при размере 2");
    EXPECT_THROWS(t.at({0, 3}), "столбец 3");
    EXPECT_THROWS(t.at({0}), "один индекс у 2D");
    EXPECT_THROWS(t.at({0, 0, 0}), "три индекса у 2D");
    EXPECT_THROWS(Tensor::ones({3}).item(), "item() у 3 элементов");
}

QA_TEST(backward_misuse, "7. Ошибки пользователя", "backward: без requires_grad, не скаляр, неверная форма grad") {
    EXPECT_THROWS(sum(Tensor::ones({2})).backward(), "нет requires_grad");
    auto x = Tensor::ones({2}).requires_grad_();
    EXPECT_THROWS((x * 2.0f).backward(), "не скаляр без grad");
    EXPECT_THROWS((x * 2.0f).backward(Tensor::ones({3})), "grad не той формы");
    EXPECT_THROWS((x * 2.0f).requires_grad_(), "requires_grad_ у не-листа");
}

QA_TEST(shape_errors, "7. Ошибки пользователя", "view/reshape/permute/expand/squeeze/unsqueeze с плохими аргументами") {
    auto x = Tensor::ones({2, 3});
    EXPECT_THROWS(x.view({4}), "view на 4 элемента из 6");
    EXPECT_THROWS(x.reshape({7}), "reshape на 7");
    EXPECT_THROWS(x.permute({0}), "permute с одной осью");
    EXPECT_THROWS(x.permute({1, 1}), "permute с повтором");
    EXPECT_THROWS(x.expand({3}), "expand в меньшую размерность");
    EXPECT_THROWS(x.expand({4, 3}), "expand оси 2 -> 4");
    EXPECT_THROWS(x.squeeze(0), "squeeze оси размера 2");
    EXPECT_THROWS(x.unsqueeze(3), "unsqueeze за концом");
    EXPECT_THROWS(x.transpose(0, 2), "transpose несуществующей оси");
    EXPECT_THROWS(sum(x, 2), "sum по оси 2");
    EXPECT_THROWS(softmax(x, 5), "softmax по оси 5");
    EXPECT_THROWS(argmax(x, 2), "argmax по оси 2");
}

QA_TEST(matmul_errors, "7. Ошибки пользователя", "matmul: 1D, 3D, несовпадающие размеры") {
    EXPECT_THROWS(matmul(Tensor::ones({3}), Tensor::ones({3, 2})), "1D слева");
    EXPECT_THROWS(matmul(Tensor::ones({2, 3}), Tensor::ones({2, 3})), "[2,3]·[2,3]");
    EXPECT_THROWS(matmul(Tensor::ones({1, 2, 3}), Tensor::ones({3, 2})), "3D");
}

QA_TEST(loss_errors, "7. Ошибки пользователя",
        "функции потерь: несовпадение форм, метки вне диапазона, отрицательные, дробные") {
    EXPECT_THROWS(nn::mse_loss(Tensor::ones({3}), Tensor::ones({3, 1})), "mse [3] vs [3,1]");
    auto logits = Tensor::zeros({2, 3});
    EXPECT_THROWS(nn::cross_entropy(logits, Tensor::from_vector({0, 3}, {2})), "класс 3 при C=3");
    EXPECT_THROWS(nn::cross_entropy(logits, Tensor::from_vector({0, -1}, {2})), "класс -1");
    EXPECT_THROWS(nn::cross_entropy(logits, Tensor::from_vector({0, 1, 2}, {3})), "3 метки на 2 примера");
    EXPECT_THROWS(nn::cross_entropy(Tensor::zeros({3}), Tensor::zeros({3})), "logits 1D");
    EXPECT_THROWS(nn::cross_entropy(logits, Tensor::from_vector({0, 1.5f}, {2})), "дробная метка 1.5");
}

QA_TEST(layer_construction_errors, "7. Ошибки пользователя",
        "Linear(0, 5), Dropout(1.0), Dropout(-0.1), оптимизатор с не-параметром") {
    EXPECT_THROWS(nn::Dropout(1.0f), "Dropout(1.0)");
    EXPECT_THROWS(nn::Dropout(-0.1f), "Dropout(-0.1)");
    EXPECT_THROWS(optim::SGD({Tensor::ones({2})}, 0.1f), "SGD с тензором без requires_grad");
    auto x = Tensor::ones({2}).requires_grad_();
    EXPECT_THROWS(optim::Adam({x * 2.0f}, 0.1f), "Adam с не-листом");
    try {
        nn::Linear bad(0, 5);
        EXPECT(false, "Linear(0, 5) отклонён");
    } catch (const qa::Failure&) {
        throw;
    } catch (const std::exception& e) {
        std::string msg = e.what();
        EXPECT(msg.find("Linear") != std::string::npos,
               "сообщение об ошибке говорит про Linear, а не про внутренности: \"" + msg + "\"");
    }
}

QA_TEST(linear_wrong_input, "7. Ошибки пользователя",
        "Linear(784,10) получил [32, 100] или вектор [784] — понятная ошибка") {
    nn::Linear layer(784, 10);
    EXPECT_THROWS(layer(Tensor::ones({32, 100})), "[32, 100]");
    EXPECT_THROWS(layer(Tensor::ones({784})), "1D вход");
}

QA_TEST(dataloader_errors, "7. Ошибки пользователя", "DataLoader: batch 0, batch за концом, разные длины") {
    data::Dataset ds{Tensor::ones({5, 2}), Tensor::ones({5})};
    EXPECT_THROWS(data::DataLoader(ds, 0), "batch_size 0");
    data::DataLoader ok(ds, 2);
    EXPECT_THROWS(ok.batch(3), "батч №3 из 3");
    EXPECT_THROWS(data::DataLoader(data::Dataset{Tensor::ones({5, 2}), Tensor::ones({4})}, 2),
                  "5 входов и 4 метки");
}
