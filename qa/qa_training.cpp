#include "qa.h"

#include <forge/forge.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <set>

using namespace forge;

namespace {

std::vector<float> values(const Tensor& t) {
    Tensor c = t.contiguous();
    return std::vector<float>(c.data(), c.data() + c.numel());
}

float train(nn::Module& model, optim::Optimizer& opt, const Tensor& x, const Tensor& y, int steps,
            bool classification) {
    float last = 0;
    for (int i = 0; i < steps; ++i) {
        opt.zero_grad();
        Tensor out = model(x);
        Tensor loss = classification ? nn::cross_entropy(out, y) : nn::mse_loss(out, y);
        loss.backward();
        opt.step();
        last = loss.item();
    }
    return last;
}

// Две "кучки" точек вокруг (-2,-2) и (2,2).
void make_blobs(std::size_t n, Tensor& x, Tensor& y) {
    auto noise = Tensor::randn({n, 2});
    x = Tensor::zeros({n, 2});
    y = Tensor::zeros({n});
    for (std::size_t i = 0; i < n; ++i) {
        float c = (i % 2 == 0) ? -2.0f : 2.0f;
        x.at({i, 0}) = c + noise.at({i, 0});
        x.at({i, 1}) = c + noise.at({i, 1});
        y.at({i}) = float(i % 2);
    }
}

// Три спирали — не разделимы прямой линией.
void make_spiral(std::size_t per_class, Tensor& x, Tensor& y) {
    std::size_t n = per_class * 3;
    x = Tensor::zeros({n, 2});
    y = Tensor::zeros({n});
    auto noise = Tensor::randn({n});
    for (std::size_t c = 0; c < 3; ++c)
        for (std::size_t i = 0; i < per_class; ++i) {
            std::size_t k = c * per_class + i;
            float r = float(i) / float(per_class);
            float t = float(c) * 4.0f + r * 4.0f + noise.at({k}) * 0.2f;
            x.at({k, 0}) = r * std::sin(t);
            x.at({k, 1}) = r * std::cos(t);
            y.at({k}) = float(c);
        }
}

void write_bytes(const std::string& path, const std::vector<unsigned char>& bytes) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(bytes.data()), long(bytes.size()));
}

std::vector<unsigned char> be32(std::uint32_t v) {
    return {static_cast<unsigned char>(v >> 24), static_cast<unsigned char>(v >> 16),
            static_cast<unsigned char>(v >> 8), static_cast<unsigned char>(v)};
}

std::vector<unsigned char> cat(std::initializer_list<std::vector<unsigned char>> parts) {
    std::vector<unsigned char> out;
    for (auto& p : parts) out.insert(out.end(), p.begin(), p.end());
    return out;
}

} // namespace

// ------------------------------------------------------------------
// Обучение на задачах с известным ответом
// ------------------------------------------------------------------

QA_TEST(linear_regression_recovers_weights, "8. Обучение",
        "y = 3x − 2 + шум: SGD находит w ≈ 3, b ≈ −2") {
    manual_seed(1);
    auto x = Tensor::randn({200, 1});
    auto y = x * 3.0f - 2.0f + Tensor::randn({200, 1}) * 0.05f;
    nn::Linear model(1, 1);
    optim::SGD opt(model.parameters(), 0.1f);
    train(model, opt, x, y, 300, false);
    EXPECT_NEAR(model.weight.item(), 3.0, 0.05, "w ≈ 3");
    EXPECT_NEAR(model.bias.item(), -2.0, 0.05, "b ≈ -2");
}

QA_TEST(multi_output_regression, "8. Обучение", "Adam восстанавливает матрицу 3×2 по данным") {
    manual_seed(2);
    auto W_true = Tensor::from_vector({1, -1, 2, 0.5f, -3, 0}, {3, 2});
    auto x = Tensor::randn({500, 3});
    auto y = matmul(x, W_true);
    nn::Linear model(3, 2, /*with_bias=*/false);
    optim::Adam opt(model.parameters(), 0.05f);
    train(model, opt, x, y, 400, false);
    auto got = values(model.weight), want = values(W_true);
    for (std::size_t i = 0; i < 6; ++i) EXPECT_NEAR(got[i], want[i], 0.02, "W[" + qa::str(i) + "]");
}

QA_TEST(blobs_classification, "8. Обучение", "две кучки точек: логистическая регрессия даёт ≥ 97%") {
    manual_seed(3);
    Tensor x, y;
    make_blobs(400, x, y);
    nn::Linear model(2, 2);
    optim::SGD opt(model.parameters(), 0.5f);
    train(model, opt, x, y, 100, true);
    NoGradGuard ng;
    EXPECT(nn::accuracy(model(x), y) >= 0.97f, "точность ≥ 97%");
}

QA_TEST(spiral_needs_hidden_layer, "8. Обучение",
        "три спирали: линейная модель < 70%, сеть со скрытым слоем ≥ 95%") {
    manual_seed(4);
    Tensor x, y;
    make_spiral(100, x, y);
    nn::Linear linear(2, 3);
    optim::Adam opt1(linear.parameters(), 0.05f);
    train(linear, opt1, x, y, 500, true);
    nn::Sequential mlp;
    mlp.add<nn::Linear>(2, 64);
    mlp.add<nn::ReLU>();
    mlp.add<nn::Linear>(64, 64);
    mlp.add<nn::ReLU>();
    mlp.add<nn::Linear>(64, 3);
    optim::Adam opt2(mlp.parameters(), 0.01f);
    train(mlp, opt2, x, y, 1500, true);
    NoGradGuard ng;
    float lin = nn::accuracy(linear(x), y), deep = nn::accuracy(mlp(x), y);
    EXPECT(lin < 0.7f, "линейная модель не справляется (" + qa::str(lin) + ")");
    EXPECT(deep >= 0.95f, "MLP справляется (" + qa::str(deep) + ")");
}

QA_TEST(momentum_helps, "8. Обучение",
        "вытянутая квадратичная функция: SGD с momentum сходится в 10+ раз лучше обычного") {
    auto run = [](float momentum) {
        auto w = Tensor::from_vector({5, 5}, {2}).requires_grad_();
        auto scale = Tensor::from_vector({1, 50}, {2});
        optim::SGD opt({w}, 0.015f, momentum);
        for (int i = 0; i < 100; ++i) {
            opt.zero_grad();
            sum(scale * w * w).backward();
            opt.step();
        }
        NoGradGuard ng;
        return sum(scale * w * w).item();
    };
    float plain = run(0.0f), fast = run(0.9f);
    EXPECT(fast < plain * 0.1f, "momentum хотя бы в 10 раз лучше (" + qa::str(plain) + " vs " + qa::str(fast) + ")");
}

QA_TEST(weight_decay_and_lr_zero, "8. Обучение",
        "weight decay сжимает веса к нулю; lr = 0 ничего не меняет") {
    auto w = Tensor::from_vector({4, -4}, {2}).requires_grad_();
    optim::SGD opt({w}, 0.1f, 0.0f, /*weight_decay=*/1.0f);
    for (int i = 0; i < 50; ++i) {
        opt.zero_grad();
        sum(w * 0.0f).backward();  // градиент данных = 0, работает только decay
        opt.step();
    }
    EXPECT(std::fabs(w.at({0})) < 0.05f && std::fabs(w.at({1})) < 0.05f, "веса ≈ 0");
    auto v = Tensor::from_vector({1, 2}, {2}).requires_grad_();
    optim::Adam frozen({v}, 0.1f);
    frozen.set_lr(0.0f);
    sum(v * v).backward();
    frozen.step();
    EXPECT(values(v) == (std::vector<float>{1, 2}), "lr = 0: веса не изменились");
}

QA_TEST(overfit_tiny_batch, "8. Обучение", "сеть запоминает 8 случайных примеров до loss < 0.01") {
    manual_seed(5);
    auto x = Tensor::randn({8, 10});
    auto y = Tensor::from_vector({0, 1, 2, 3, 0, 1, 2, 3}, {8});
    nn::Sequential m;
    m.add<nn::Linear>(10, 32);
    m.add<nn::Tanh>();
    m.add<nn::Linear>(32, 4);
    optim::Adam opt(m.parameters(), 0.01f);
    EXPECT(train(m, opt, x, y, 400, true) < 0.01f, "loss < 0.01");
}

QA_TEST(dropout_eval_is_deterministic, "8. Обучение",
        "в eval() Dropout выключен: два прогона дают одинаковый ответ, в train() — разный") {
    manual_seed(6);
    nn::Sequential m;
    m.add<nn::Linear>(4, 16);
    m.add<nn::Dropout>(0.5f);
    m.add<nn::Linear>(16, 2);
    auto x = Tensor::randn({3, 4});
    NoGradGuard ng;
    EXPECT(values(m(x)) != values(m(x)), "в train() выходы различаются");
    m.eval();
    EXPECT(values(m(x)) == values(m(x)), "в eval() выходы одинаковые");
    EXPECT(!m.is_training(), "is_training() == false");
}

QA_TEST(mnist_one_epoch, "8. Обучение",
        "настоящий MNIST (если скачан в data/): 1 эпоха даёт ≥ 93% на test") {
    std::ifstream probe("data/train-images-idx3-ubyte");
    if (!probe) {
        std::printf("MNIST не найден в data/ — пропуск");
        return;
    }
    auto train_set = data::load_mnist("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte");
    auto test_set = data::load_mnist("data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte");
    manual_seed(0);
    nn::Sequential m;
    m.add<nn::Linear>(784, 128);
    m.add<nn::ReLU>();
    m.add<nn::Linear>(128, 10);
    optim::Adam opt(m.parameters(), 1e-3f);
    data::DataLoader loader(train_set, 64);
    for (std::size_t b = 0; b < loader.num_batches(); ++b) {
        auto [x, y] = loader.batch(b);
        opt.zero_grad();
        nn::cross_entropy(m(x), y).backward();
        opt.step();
    }
    NoGradGuard ng;
    float acc = nn::accuracy(m(test_set.inputs), test_set.targets);
    EXPECT(acc >= 0.93f, "точность ≥ 93% (получено " + qa::str(acc) + ")");
}

// ------------------------------------------------------------------
// Сохранение и загрузка
// ------------------------------------------------------------------

QA_TEST(save_load_same_predictions, "9. Файлы", "сохранили → загрузили в новую модель → те же предсказания") {
    manual_seed(7);
    auto make = [] {
        auto m = std::make_shared<nn::Sequential>();
        m->add<nn::Linear>(5, 8);
        m->add<nn::ReLU>();
        m->add<nn::Linear>(8, 3);
        return m;
    };
    auto a = make();
    auto x = Tensor::randn({4, 5});
    a->save("qa_model.bin");
    auto b = make();
    b->load("qa_model.bin");
    std::remove("qa_model.bin");
    NoGradGuard ng;
    EXPECT(values((*a)(x)) == values((*b)(x)), "предсказания совпадают бит в бит");
}

QA_TEST(failed_load_leaves_model_intact, "9. Файлы",
        "загрузка весов другой архитектуры: ошибка, и модель НЕ испорчена наполовину") {
    manual_seed(8);
    nn::Sequential a;
    a.add<nn::Linear>(3, 4);
    a.add<nn::Linear>(4, 2);
    a.save("qa_a.bin");
    nn::Sequential b;
    b.add<nn::Linear>(3, 4);
    b.add<nn::Linear>(4, 5);  // второй слой отличается
    auto before = values(b.parameters()[0]);
    EXPECT_THROWS(b.load("qa_a.bin"), "несовпадающая архитектура");
    std::remove("qa_a.bin");
    EXPECT(values(b.parameters()[0]) == before, "первый слой остался прежним");
}

QA_TEST(corrupted_weight_files, "9. Файлы",
        "пустой, мусорный, обрезанный файл, огромная длина имени, нет файла, нет папки") {
    nn::Linear m(3, 2);
    m.save("qa_ok.bin");
    std::ifstream in("qa_ok.bin", std::ios::binary);
    std::vector<unsigned char> good((std::istreambuf_iterator<char>(in)), {});
    write_bytes("qa_empty.bin", {});
    write_bytes("qa_garbage.bin", {'h', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd'});
    write_bytes("qa_trunc.bin", std::vector<unsigned char>(good.begin(), good.end() - 5));
    auto huge = good;
    for (int i = 12; i < 20; ++i) huge[std::size_t(i)] = 0x7f;  // длина имени ~ 9e18
    write_bytes("qa_huge.bin", huge);
    EXPECT_THROWS(m.load("qa_empty.bin"), "пустой файл");
    EXPECT_THROWS(m.load("qa_garbage.bin"), "мусор");
    EXPECT_THROWS(m.load("qa_trunc.bin"), "обрезанный");
    EXPECT_THROWS(m.load("qa_huge.bin"), "огромная длина имени");
    EXPECT_THROWS(m.load("qa_no_such_file.bin"), "нет файла");
    EXPECT_THROWS(m.save("no_such_dir/x/y.bin"), "нет папки");
    for (const char* f : {"qa_ok.bin", "qa_empty.bin", "qa_garbage.bin", "qa_trunc.bin", "qa_huge.bin"})
        std::remove(f);
}

// ------------------------------------------------------------------
// DataLoader
// ------------------------------------------------------------------

QA_TEST(dataloader_epoch_covers_everything, "10. DataLoader",
        "за эпоху каждый пример встречается ровно один раз; последний батч неполный") {
    data::Dataset ds{Tensor::arange(0, 103).reshape({103, 1}), Tensor::arange(0, 103)};
    data::DataLoader loader(ds, 10, true, 42);
    EXPECT(loader.num_batches() == 11, "11 батчей");
    for (int epoch = 0; epoch < 3; ++epoch) {
        loader.reshuffle();
        std::multiset<float> seen;
        for (std::size_t b = 0; b < loader.num_batches(); ++b) {
            auto [x, y] = loader.batch(b);
            if (b == 10) EXPECT(x.shape()[0] == 3, "последний батч из 3");
            for (std::size_t i = 0; i < x.shape()[0]; ++i) {
                EXPECT(x.at({i, 0}) == y.at({i}), "вход и метка не перепутаны");
                seen.insert(y.at({i}));
            }
        }
        EXPECT(seen.size() == 103 && std::set<float>(seen.begin(), seen.end()).size() == 103,
               "103 уникальных примера");
    }
}

QA_TEST(dataloader_shuffle_semantics, "10. DataLoader",
        "shuffle=false — по порядку; одинаковый seed — одинаково; reshuffle меняет порядок") {
    data::Dataset ds{Tensor::arange(0, 50).reshape({50, 1}), Tensor::arange(0, 50)};
    data::DataLoader plain(ds, 50, false);
    EXPECT(values(plain.batch(0).second) == values(Tensor::arange(0, 50)), "без перемешивания");
    data::DataLoader a(ds, 50, true, 1), b(ds, 50, true, 1), c(ds, 50, true, 2);
    EXPECT(values(a.batch(0).second) == values(b.batch(0).second), "seed 1 == seed 1");
    EXPECT(values(a.batch(0).second) != values(c.batch(0).second), "seed 1 != seed 2");
    auto before = values(a.batch(0).second);
    a.reshuffle();
    EXPECT(values(a.batch(0).second) != before, "reshuffle меняет порядок");
}

QA_TEST(dataloader_edge_sizes, "10. DataLoader", "batch больше датасета; 3D-входы; пустой датасет") {
    data::Dataset small{Tensor::ones({3, 2}), Tensor::zeros({3})};
    data::DataLoader big(small, 100);
    EXPECT(big.num_batches() == 1 && big.batch(0).first.shape()[0] == 3, "1 батч из 3");
    data::Dataset images{Tensor::arange(0, 24).reshape({6, 2, 2}), Tensor::arange(0, 6)};
    data::DataLoader l(images, 4, false);
    EXPECT(l.batch(1).first.shape() == (Shape{2, 2, 2}), "форма [2, 2, 2]");
    EXPECT(l.batch(1).first.at({1, 1, 1}) == 23.0f, "последний пиксель последней картинки");
    data::Dataset empty{Tensor::zeros({0, 3}), Tensor::zeros({0})};
    data::DataLoader e(empty, 4);
    EXPECT(e.num_batches() == 0, "0 батчей");
}

// ------------------------------------------------------------------
// MNIST-загрузчик на подделанных файлах
// ------------------------------------------------------------------

QA_TEST(mnist_loader_tiny_valid_file, "11. MNIST-файлы",
        "самодельный IDX из 3 картинок 2×2: значения делятся на 255, форма [3, 4]") {
    write_bytes("qa_img", cat({be32(2051), be32(3), be32(2), be32(2),
                               {0, 255, 51, 102, 1, 2, 3, 4, 255, 255, 255, 255}}));
    write_bytes("qa_lbl", cat({be32(2049), be32(3), {7, 0, 9}}));
    auto ds = data::load_mnist("qa_img", "qa_lbl");
    EXPECT(ds.inputs.shape() == (Shape{3, 4}), "форма [3, 4]");
    EXPECT_NEAR(ds.inputs.at({0, 2}), 0.2, 1e-6, "51/255 = 0.2");
    EXPECT(ds.targets.at({2}) == 9.0f, "третья метка 9");
    std::remove("qa_img");
    std::remove("qa_lbl");
}

QA_TEST(mnist_loader_bad_files, "11. MNIST-файлы",
        "нет файла, перепутаны images/labels, обрезан, число меток не совпадает") {
    write_bytes("qa_img", cat({be32(2051), be32(2), be32(2), be32(2), {1, 2, 3, 4, 5, 6, 7, 8}}));
    write_bytes("qa_lbl", cat({be32(2049), be32(2), {1, 2}}));
    write_bytes("qa_lbl3", cat({be32(2049), be32(3), {1, 2, 3}}));
    write_bytes("qa_img_short", cat({be32(2051), be32(2), be32(2), be32(2), {1, 2, 3}}));
    EXPECT_THROWS(data::load_mnist("qa_nothing", "qa_lbl"), "нет файла картинок");
    EXPECT_THROWS(data::load_mnist("qa_lbl", "qa_img"), "файлы перепутаны местами");
    EXPECT_THROWS(data::load_mnist("qa_img_short", "qa_lbl"), "обрезанные картинки");
    EXPECT_THROWS(data::load_mnist("qa_img", "qa_lbl3"), "2 картинки и 3 метки");
    for (const char* f : {"qa_img", "qa_lbl", "qa_lbl3", "qa_img_short"}) std::remove(f);
}
