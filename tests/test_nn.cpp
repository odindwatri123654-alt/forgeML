#include "test_framework.h"

#include <forge/forge.h>

#include <cstdio>

using namespace forge;

TEST(nn_linear_shapes_and_parameters) {
    nn::Linear layer(4, 3);
    CHECK(layer.weight.shape() == (Shape{4, 3}));
    CHECK(layer.bias.shape() == (Shape{3}));
    CHECK(layer.parameters().size() == 2);
    CHECK(layer.num_parameters() == 15);
    CHECK(layer(Tensor::ones({5, 4})).shape() == (Shape{5, 3}));
    nn::Linear no_bias(4, 3, false);
    CHECK(no_bias.parameters().size() == 1);
}

TEST(nn_sequential_names) {
    nn::Sequential model;
    model.add<nn::Linear>(2, 4);
    model.add<nn::ReLU>();
    model.add<nn::Linear>(4, 1);
    auto named = model.named_parameters();
    CHECK(named.size() == 4);
    CHECK(named[0].first == "0.weight");
    CHECK(named[3].first == "2.bias");
}

TEST(nn_losses) {
    auto pred = Tensor::from_vector({1, 2, 3}, {3});
    auto target = Tensor::from_vector({1, 2, 5}, {3});
    CHECK_NEAR(nn::mse_loss(pred, target).item(), 4.0 / 3.0, 1e-5);
    CHECK_THROWS(nn::mse_loss(pred, Tensor::ones({3, 1})));

    // равные логиты для 4 классов -> loss = ln(4)
    auto logits = Tensor::zeros({2, 4});
    auto labels = Tensor::from_vector({0, 3}, {2});
    CHECK_NEAR(nn::cross_entropy(logits, labels).item(), std::log(4.0), 1e-5);
    CHECK_THROWS(nn::cross_entropy(logits, Tensor::from_vector({0, 4}, {2})));

    auto confident = Tensor::from_vector({5, 0, 0, 0, 0, 5}, {2, 3});
    CHECK(nn::accuracy(confident, Tensor::from_vector({0, 2}, {2})) == 1.0f);
    CHECK(nn::accuracy(confident, Tensor::from_vector({1, 2}, {2})) == 0.5f);
}

TEST(nn_dropout_train_eval) {
    nn::Dropout drop(0.5f);
    auto x = Tensor::ones({1000});
    auto y = drop(x);
    std::size_t zeros = 0;
    for (std::size_t i = 0; i < 1000; ++i) {
        float v = y.at({i});
        CHECK(v == 0.0f || v == 2.0f);
        zeros += v == 0.0f;
    }
    CHECK(zeros > 400 && zeros < 600);
    drop.eval();
    CHECK(drop(x).at({0}) == 1.0f);
}

TEST(optim_sgd_step) {
    auto w = Tensor::from_vector({1, 2}, {2}).requires_grad_();
    optim::SGD opt({w}, 0.1f);
    sum(w * w).backward();  // grad = 2w = [2, 4]
    opt.step();
    CHECK_NEAR(w.at({0}), 0.8, 1e-6);
    CHECK_NEAR(w.at({1}), 1.6, 1e-6);
    opt.zero_grad();
    CHECK(!w.grad().defined());
}

TEST(optim_adam_first_step_is_lr) {
    // на первом шаге Adam сдвигает каждый параметр ровно на lr (по знаку градиента)
    auto w = Tensor::from_vector({1, -1}, {2}).requires_grad_();
    optim::Adam opt({w}, 0.1f);
    sum(w * 3.0f).backward();
    opt.step();
    CHECK_NEAR(w.at({0}), 0.9, 1e-5);
    CHECK_NEAR(w.at({1}), -1.1, 1e-5);
}

TEST(train_xor) {
    manual_seed(0);
    auto x = Tensor::from_vector({0, 0, 0, 1, 1, 0, 1, 1}, {4, 2});
    auto y = Tensor::from_vector({0, 1, 1, 0}, {4, 1});
    nn::Sequential model;
    model.add<nn::Linear>(2, 8);
    model.add<nn::Tanh>();
    model.add<nn::Linear>(8, 1);
    model.add<nn::Sigmoid>();
    optim::Adam opt(model.parameters(), 0.05f);
    float loss_value = 1.0f;
    for (int step = 0; step < 500; ++step) {
        opt.zero_grad();
        Tensor loss = nn::mse_loss(model(x), y);
        loss.backward();
        opt.step();
        loss_value = loss.item();
    }
    CHECK(loss_value < 0.01f);
}

TEST(save_and_load_roundtrip) {
    manual_seed(1);
    nn::Linear a(3, 2);
    const std::string path = "forge_test_weights.bin";
    a.save(path);
    manual_seed(2);
    nn::Linear b(3, 2);
    CHECK(b.weight.at({0, 0}) != a.weight.at({0, 0}));
    b.load(path);
    CHECK(b.weight.at({0, 0}) == a.weight.at({0, 0}));
    CHECK(b.bias.at({1}) == a.bias.at({1}));
    nn::Linear wrong(4, 2);
    CHECK_THROWS(wrong.load(path));
    std::remove(path.c_str());
}

TEST(dataloader_batches) {
    data::Dataset ds{Tensor::arange(0, 10).view({5, 2}), Tensor::arange(0, 5)};
    data::DataLoader loader(ds, 2, /*shuffle=*/false);
    CHECK(loader.num_batches() == 3);
    auto [x, y] = loader.batch(2);
    CHECK(x.shape() == (Shape{1, 2}));
    CHECK(x.at({0, 1}) == 9.0f);
    CHECK(y.at({0}) == 4.0f);

    data::DataLoader shuffled(ds, 5, /*shuffle=*/true, 123);
    auto [sx, sy] = shuffled.batch(0);
    for (std::size_t i = 0; i < 5; ++i) {
        CHECK(sx.at({i, 0}) == 2.0f * sy.at({i}));  // строки и метки перемешаны вместе
    }
}
