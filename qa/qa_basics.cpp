#include "qa.h"

#include <forge/forge.h>

#include <limits>
#include <random>
#include <sstream>

using namespace forge;

namespace {

// Эталонный matmul — "в лоб", в double.
double naive_dot(const Tensor& a, const Tensor& b, std::size_t i, std::size_t j) {
    double s = 0;
    for (std::size_t k = 0; k < a.shape()[1]; ++k) s += double(a.at({i, k})) * b.at({k, j});
    return s;
}

std::vector<float> values(const Tensor& t) {
    Tensor c = t.contiguous();
    return std::vector<float>(c.data(), c.data() + c.numel());
}

} // namespace

// ------------------------------------------------------------------
// Знакомство: примеры из README
// ------------------------------------------------------------------

QA_TEST(readme_tensor_basics, "1. Знакомство по README",
        "примеры тензоров из README дают обещанные значения") {
    auto b = Tensor::from_vector({1, 2, 3, 4, 5, 6}, {2, 3});
    EXPECT(b.at({1, 0}) == 4.0f, "b.at({1,0}) == 4");
    b.at({1, 0}) = 10;
    EXPECT(b.shape() == (Shape{2, 3}) && b.numel() == 6, "shape {2,3}, numel 6");
    EXPECT(Tensor::full({}, 5).item() == 5.0f, "item() скаляра == 5");
    Tensor c = b;
    c.at({0, 0}) = 99;
    EXPECT(b.at({0, 0}) == 99.0f, "копия Tensor — тот же тензор");
    Tensor d = b.clone();
    d.at({0, 0}) = -1;
    EXPECT(b.at({0, 0}) == 99.0f, "clone() — независимая копия");
}

QA_TEST(readme_quickstart_training_step, "1. Знакомство по README",
        "фрагмент из начала README: шаги обучения уменьшают loss") {
    manual_seed(1);
    nn::Sequential model;
    model.add<nn::Linear>(784, 128);
    model.add<nn::ReLU>();
    model.add<nn::Linear>(128, 10);
    optim::Adam optimizer(model.parameters(), 1e-3f);
    Tensor images = Tensor::rand({32, 784});
    Tensor labels = Tensor::zeros({32});
    for (std::size_t i = 0; i < 32; ++i) labels.at({i}) = float(i % 10);
    float first = 0, last = 0;
    for (int step = 0; step < 30; ++step) {
        optimizer.zero_grad();
        Tensor loss = nn::cross_entropy(model(images), labels);
        loss.backward();
        optimizer.step();
        if (step == 0) first = loss.item();
        last = loss.item();
    }
    EXPECT(std::isfinite(last), "loss конечен");
    EXPECT(last < first * 0.5f, "loss упал хотя бы вдвое за 30 шагов (" + qa::str(first) +
                                    " -> " + qa::str(last) + ")");
}

QA_TEST(readme_custom_module, "1. Знакомство по README",
        "свой Module из README собирается и считает параметры") {
    struct MLP : nn::Module {
        std::shared_ptr<nn::Linear> fc1 = std::make_shared<nn::Linear>(784, 256);
        std::shared_ptr<nn::Linear> fc2 = std::make_shared<nn::Linear>(256, 10);
        MLP() {
            register_module("fc1", fc1);
            register_module("fc2", fc2);
        }
        Tensor forward(const Tensor& x) override { return (*fc2)(relu((*fc1)(x))); }
    } model;
    EXPECT(model.num_parameters() == 784 * 256 + 256 + 256 * 10 + 10, "203530 параметров");
    EXPECT(model.named_parameters()[0].first == "fc1.weight", "имя первого параметра fc1.weight");
    EXPECT(model(Tensor::rand({3, 784})).shape() == (Shape{3, 10}), "выход [3, 10]");
}

// ------------------------------------------------------------------
// Создание тензоров и граничные случаи
// ------------------------------------------------------------------

QA_TEST(arange_like_numpy, "2. Создание тензоров",
        "arange ведёт себя как numpy: дробный шаг, отрицательный шаг, пустой диапазон") {
    EXPECT(Tensor::arange(0, 1, 0.1f).numel() == 10, "arange(0,1,0.1) — 10 элементов");
    EXPECT(Tensor::arange(0, 0.3f, 0.1f).numel() == 3, "arange(0,0.3,0.1) — 3 элемента");
    auto down = Tensor::arange(5, 0, -1);
    EXPECT(values(down) == (std::vector<float>{5, 4, 3, 2, 1}), "arange(5,0,-1) = 5 4 3 2 1");
    EXPECT(Tensor::arange(0, 0).numel() == 0, "arange(0,0) пустой");
    EXPECT(Tensor::arange(1, 0).numel() == 0, "arange(1,0) пустой");
    EXPECT(Tensor::arange(0, 2, 0.5f).at({3}) == 1.5f, "arange(0,2,0.5)[3] == 1.5");
}

QA_TEST(empty_tensors, "2. Создание тензоров",
        "тензоры с нулевым размером: создание, печать, sum, matmul не падают") {
    auto e = Tensor::zeros({0, 3});
    EXPECT(e.numel() == 0, "numel == 0");
    EXPECT(sum(e).item() == 0.0f, "sum пустого == 0");
    EXPECT(sum(e, 0).shape() == (Shape{3}), "sum по пустой оси -> [3]");
    std::ostringstream os;
    os << e << Tensor::zeros({0});
    auto m = matmul(Tensor::ones({2, 0}), Tensor::ones({0, 3}));
    EXPECT(m.shape() == (Shape{2, 3}) && sum(m).item() == 0.0f, "[2,0]·[0,3] = нули [2,3]");
    EXPECT(Tensor::eye(0).numel() == 0, "eye(0) пустая");
    EXPECT(Tensor::randn({0}).numel() == 0, "randn({0}) пустой");
}

QA_TEST(empty_tensor_arithmetic, "2. Создание тензоров",
        "арифметика с пустым тензором: zeros({0,3}) * 2, + bias, relu — форма [0, 3]") {
    auto e = Tensor::zeros({0, 3});
    EXPECT((e * 2.0f).shape() == (Shape{0, 3}), "zeros({0,3}) * 2 -> [0, 3]");
    EXPECT((e + Tensor::ones({3})).shape() == (Shape{0, 3}), "zeros({0,3}) + bias[3] -> [0, 3]");
    EXPECT(relu(e).shape() == (Shape{0, 3}), "relu(zeros({0,3})) -> [0, 3]");
}

QA_TEST(scalar_arithmetic, "2. Создание тензоров", "арифметика скаляров (shape {})") {
    auto a = Tensor::full({}, 2.0f);
    auto b = Tensor::full({}, 3.0f);
    auto c = a * b + 1.0f;
    EXPECT(c.ndim() == 0, "результат — скаляр");
    EXPECT(c.item() == 7.0f, "2*3+1 == 7");
    EXPECT((a / b * b).item() == 2.0f, "a/b*b == a");
}

QA_TEST(random_statistics, "2. Создание тензоров",
        "randn ~ N(0,1), rand ~ U[0,1): среднее и разброс по 200 000 чисел") {
    manual_seed(123);
    auto n = Tensor::randn({200000});
    double m = mean(n).item();
    double var = mean(pow(n - float(m), 2)).item();
    EXPECT_NEAR(m, 0.0, 0.01, "среднее randn ≈ 0");
    EXPECT_NEAR(var, 1.0, 0.02, "дисперсия randn ≈ 1");
    auto u = Tensor::rand({200000});
    EXPECT_NEAR(mean(u).item(), 0.5, 0.01, "среднее rand ≈ 0.5");
    float lo = -max(-u, 0).item(), hi = max(u, 0).item();
    EXPECT(lo >= 0.0f && hi < 1.0f, "rand в [0, 1)");
}

QA_TEST(seed_reproducibility, "2. Создание тензоров",
        "manual_seed делает одинаковыми и randn, и инициализацию слоёв") {
    manual_seed(5);
    auto a = Tensor::randn({10});
    nn::Linear la(4, 4);
    manual_seed(5);
    auto b = Tensor::randn({10});
    nn::Linear lb(4, 4);
    EXPECT(values(a) == values(b), "randn совпадает");
    EXPECT(values(la.weight) == values(lb.weight), "веса Linear совпадают");
}

QA_TEST(big_tensor_10m, "2. Создание тензоров",
        "тензор на 10 млн элементов: создание, сумма, среднее") {
    auto t = Tensor::ones({10000, 1000});
    EXPECT(sum(t).item() == 1e7f, "сумма 10 млн единиц");
    EXPECT_NEAR(mean(t * 0.1f).item(), 0.1, 1e-6, "среднее 0.1");
}

QA_TEST(six_dim_tensor, "2. Создание тензоров", "6-мерный тензор: permute, sum по осям, reshape") {
    auto t = Tensor::arange(0, 720).reshape({1, 2, 3, 4, 5, 6});
    EXPECT(t.at({0, 1, 2, 3, 4, 5}) == 719.0f, "последний элемент 719");
    auto p = t.permute({5, 4, 3, 2, 1, 0});
    EXPECT(p.at({5, 4, 3, 2, 1, 0}) == 719.0f, "permute сохраняет элемент");
    EXPECT(sum(sum(p, 0), 0).shape() == (Shape{4, 3, 2, 1}), "две редукции подряд");
    EXPECT(sum(p).item() == 719.0f * 720.0f / 2.0f, "сумма 0..719");
}

// ------------------------------------------------------------------
// Математика против ручного расчёта
// ------------------------------------------------------------------

QA_TEST(matmul_many_shapes, "3. Математика",
        "matmul против наивного цикла: 1x1, 1xN, Nx1, нечётные размеры, транспонированные входы") {
    manual_seed(9);
    struct S { std::size_t n, k, m; };
    for (S s : {S{1, 1, 1}, S{1, 7, 1}, S{5, 1, 3}, S{1, 64, 33}, S{257, 129, 65}, S{3, 300, 2}}) {
        auto a = Tensor::randn({s.n, s.k});
        auto b = Tensor::randn({s.k, s.m});
        auto c = matmul(a, b);
        for (std::size_t i = 0; i < s.n; i += std::max<std::size_t>(1, s.n / 5))
            for (std::size_t j = 0; j < s.m; j += std::max<std::size_t>(1, s.m / 5))
                EXPECT_NEAR(c.at({i, j}), naive_dot(a, b, i, j), 1e-3, "элемент matmul");
    }
    auto a = Tensor::randn({6, 4});
    auto b = Tensor::randn({6, 5});
    auto c = matmul(a.transpose(0, 1), b);  // [4,6]·[6,5]
    auto ac = a.transpose(0, 1).contiguous();
    EXPECT_NEAR(c.at({3, 4}), naive_dot(ac, b, 3, 4), 1e-4, "matmul с транспонированным входом");
}

QA_TEST(reductions_every_dim, "3. Математика",
        "sum/mean/max по каждой оси 3D-тензора против ручного подсчёта") {
    manual_seed(2);
    auto x = Tensor::randn({3, 4, 5});
    for (std::size_t d = 0; d < 3; ++d) {
        auto s = sum(x, d, true);
        auto mx = max(x, d, true);
        auto me = mean(x, d);
        for (std::size_t i = 0; i < 3; ++i)
            for (std::size_t j = 0; j < 4; ++j)
                for (std::size_t k = 0; k < 5; ++k) {
                    Shape idx{i, j, k};
                    if (idx[d] != 0) continue;
                    double ref = 0, best = -1e30;
                    for (std::size_t t = 0; t < x.shape()[d]; ++t) {
                        Shape q = idx;
                        q[d] = t;
                        ref += x.at(q);
                        best = std::max(best, double(x.at(q)));
                    }
                    EXPECT_NEAR(s.at(idx), ref, 1e-4, "sum по оси " + qa::str(d));
                    EXPECT_NEAR(mx.at(idx), best, 0, "max по оси " + qa::str(d));
                    Shape r = idx;
                    r.erase(r.begin() + long(d));
                    EXPECT_NEAR(me.at(r), ref / double(x.shape()[d]), 1e-5, "mean по оси");
                }
    }
}

QA_TEST(softmax_properties, "3. Математика",
        "softmax: строки суммируются в 1, не зависит от сдвига, не переполняется на 1e4") {
    manual_seed(4);
    auto x = Tensor::randn({8, 6}) * 5.0f;
    auto s = softmax(x, 1);
    auto rows = sum(s, 1);
    for (std::size_t i = 0; i < 8; ++i) EXPECT_NEAR(rows.at({i}), 1.0, 1e-5, "сумма строки = 1");
    auto shifted = values(softmax(x + 1000.0f, 1));
    auto plain = values(s);
    for (std::size_t i = 0; i < plain.size(); ++i)
        EXPECT_NEAR(shifted[i], plain[i], 1e-5, "softmax(x + 1000) == softmax(x)");
    auto huge = Tensor::from_vector({1e4f, 0, -1e4f}, {1, 3});
    auto p = values(softmax(huge, 1));
    EXPECT(std::isfinite(p[0]) && p[0] == 1.0f && p[2] == 0.0f, "softmax([1e4,0,-1e4]) = [1,0,0]");
    auto lp = values(log_softmax(huge, 1));
    EXPECT(std::isfinite(lp[1]) && lp[0] == 0.0f, "log_softmax конечен на 1e4");
}

QA_TEST(activations_extreme_inputs, "3. Математика",
        "sigmoid/tanh/exp на очень больших и маленьких числах: без NaN") {
    auto x = Tensor::from_vector({-1000, -100, 0, 100, 1000}, {5});
    auto s = values(sigmoid(x));
    EXPECT(s[0] == 0.0f && s[2] == 0.5f && s[4] == 1.0f, "sigmoid(±1000) = 0 / 1");
    for (float v : s) EXPECT(!std::isnan(v), "sigmoid без NaN");
    auto t = values(tanh(x));
    EXPECT(t[0] == -1.0f && t[4] == 1.0f, "tanh(±1000) = ±1");
    EXPECT(std::isinf(values(exp(x))[4]), "exp(1000) = inf (как в IEEE/PyTorch)");
}

QA_TEST(ieee_special_values, "3. Математика",
        "log(0), 1/0, 0/0 дают -inf, inf, nan без исключений (как PyTorch)") {
    auto z = Tensor::zeros({1});
    EXPECT(std::isinf(log(z).item()) && log(z).item() < 0, "log(0) = -inf");
    EXPECT(std::isinf((1.0f / z).item()), "1/0 = inf");
    EXPECT(std::isnan((z / z).item()), "0/0 = nan");
    EXPECT(std::isnan(mean(Tensor::zeros({0})).item()), "mean пустого = nan");
}

QA_TEST(nan_propagation_in_max, "3. Математика",
        "max по оси с NaN внутри: ожидаем NaN (так делает PyTorch)") {
    float nan = std::numeric_limits<float>::quiet_NaN();
    auto x = Tensor::from_vector({1, nan, 3}, {1, 3});
    EXPECT(std::isnan(max(x, 1).item()), "max([1, nan, 3]) = nan");
}

QA_TEST(long_sum_precision, "3. Математика",
        "сумма 10 млн чисел 0.1: и sum(x), и sum(x, dim) должны дать ≈ 1 000 000") {
    auto x = Tensor::full({1, 10000000}, 0.1f);
    EXPECT_NEAR(sum(x).item(), 1e6, 1.0, "sum(x) ≈ 1e6");
    EXPECT_NEAR(sum(x, 1).item(), 1e6, 1000.0, "sum(x, 1) ≈ 1e6 (допуск 0.1%)");
    EXPECT_NEAR(mean(x, 1).item(), 0.1, 1e-4, "mean(x, 1) ≈ 0.1");
}

// ------------------------------------------------------------------
// Broadcasting
// ------------------------------------------------------------------

QA_TEST(broadcast_3d, "4. Broadcasting", "[2,1,3] + [4,1] -> [2,4,3] с правильными значениями") {
    auto a = Tensor::arange(0, 6).reshape({2, 1, 3});
    auto b = Tensor::arange(0, 4).reshape({4, 1}) * 10.0f;
    auto c = a + b;
    EXPECT(c.shape() == (Shape{2, 4, 3}), "форма [2,4,3]");
    EXPECT(c.at({1, 3, 2}) == 5.0f + 30.0f, "c[1][3][2] = a[1][0][2] + b[3][0]");
}

QA_TEST(broadcast_error_message, "4. Broadcasting",
        "несовместимые формы: исключение с понятным текстом") {
    try {
        Tensor::ones({2, 3}) + Tensor::ones({2});
        EXPECT(false, "исключение");
    } catch (const std::invalid_argument& e) {
        std::string msg = e.what();
        EXPECT(msg.find("[2, 3]") != std::string::npos && msg.find("[2]") != std::string::npos,
               "в сообщении видны обе формы: " + msg);
    }
}

QA_TEST(broadcast_scalar_everywhere, "4. Broadcasting", "скаляр складывается с тензором любой формы") {
    auto s = Tensor::full({}, 2.0f);
    for (Shape shape : {Shape{}, Shape{3}, Shape{2, 3}, Shape{2, 1, 4}}) {
        auto r = Tensor::ones(shape) * s;
        EXPECT(r.shape() == shape && sum(r).item() == 2.0f * float(numel(shape)), "форма сохраняется");
    }
}

// ------------------------------------------------------------------
// Views и общая память
// ------------------------------------------------------------------

QA_TEST(views_share_memory, "5. Views",
        "view/transpose/expand смотрят в ту же память; reshape не-contiguous — копия") {
    auto base = Tensor::arange(0, 6);
    auto m = base.view({2, 3});
    m.at({1, 2}) = 100;
    EXPECT(base.at({5}) == 100.0f, "изменение через view видно в оригинале");
    auto t = m.transpose(0, 1);
    EXPECT(values(t.transpose(0, 1)) == values(m), "transpose дважды = исходный");
    auto r = t.reshape({6});
    r.at({0}) = -7;
    EXPECT(base.at({0}) == 0.0f, "reshape транспонированного — копия, оригинал не тронут");
    auto e = Tensor::from_vector({1, 2, 3}, {3}).expand({4, 3});
    e.at({3, 1}) = 50;
    EXPECT(e.at({0, 1}) == 50.0f, "у expand все строки — одни и те же ячейки");
}

QA_TEST(contiguous_and_detach_no_copy, "5. Views",
        "contiguous() у contiguous-тензора и detach() не копируют данные") {
    auto a = Tensor::randn({4, 4});
    EXPECT(a.contiguous().data() == a.data(), "contiguous() — тот же буфер");
    EXPECT(a.detach().data() == a.data(), "detach() — тот же буфер");
    EXPECT(a.transpose(0, 1).contiguous().data() != a.data(), "contiguous() транспонированного — новый буфер");
}

QA_TEST(permute_roundtrip, "5. Views", "permute и обратная перестановка возвращают исходный тензор") {
    auto x = Tensor::randn({2, 3, 4});
    auto y = x.permute({1, 2, 0}).permute({2, 0, 1});
    EXPECT(y.shape() == x.shape() && values(y) == values(x), "x.permute(p).permute(p^-1) == x");
    EXPECT(values(x.transpose(1, 1)) == values(x), "transpose(1,1) ничего не меняет");
}
