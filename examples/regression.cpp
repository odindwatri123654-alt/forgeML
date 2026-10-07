// День 5: регрессия — сеть приближает функцию sin(x) на отрезке [-3, 3].
#include <forge/forge.h>

#include <cmath>
#include <iostream>

using namespace forge;

int main() {
    enable_utf8_console();
    manual_seed(0);
    const std::size_t n = 200;
    Tensor x = Tensor::arange(-3.0f, 3.0f, 6.0f / n).view({n, 1});
    Tensor y = Tensor::zeros({n, 1});
    for (std::size_t i = 0; i < n; ++i) {
        y.at({i, 0}) = std::sin(x.at({i, 0}));
    }

    nn::Sequential model;
    model.add<nn::Linear>(1, 32);
    model.add<nn::Tanh>();
    model.add<nn::Linear>(32, 32);
    model.add<nn::Tanh>();
    model.add<nn::Linear>(32, 1);

    optim::Adam optimizer(model.parameters(), 0.01f);
    for (int epoch = 0; epoch <= 1000; ++epoch) {
        optimizer.zero_grad();
        Tensor loss = nn::mse_loss(model(x), y);
        loss.backward();
        optimizer.step();
        if (epoch % 200 == 0) {
            std::cout << "epoch " << epoch << "  mse " << loss.item() << "\n";
        }
    }

    NoGradGuard no_grad;
    std::cout << "\n   x      sin(x)   model(x)\n";
    for (float v : {-2.5f, -1.0f, 0.0f, 1.0f, 2.5f}) {
        float p = model(Tensor::full({1, 1}, v)).item();
        std::cout << "  " << v << "\t" << std::sin(v) << "\t" << p << "\n";
    }
}
