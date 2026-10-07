// День 5: сеть выучивает XOR — функцию, которую не может выучить один линейный слой.
#include <forge/forge.h>

#include <iostream>

using namespace forge;

int main() {
    enable_utf8_console();
    manual_seed(0);
    auto x = Tensor::from_vector({0, 0, 0, 1, 1, 0, 1, 1}, {4, 2});
    auto y = Tensor::from_vector({0, 1, 1, 0}, {4, 1});

    nn::Sequential model;
    model.add<nn::Linear>(2, 8);
    model.add<nn::Tanh>();
    model.add<nn::Linear>(8, 1);
    model.add<nn::Sigmoid>();

    optim::Adam optimizer(model.parameters(), 0.05f);

    for (int step = 0; step <= 500; ++step) {
        optimizer.zero_grad();               // 1. забыть старые градиенты
        Tensor prediction = model(x);        // 2. прямой проход
        Tensor loss = nn::mse_loss(prediction, y);
        loss.backward();                     // 3. обратный проход
        optimizer.step();                    // 4. обновить веса
        if (step % 100 == 0) {
            std::cout << "step " << step << "  loss " << loss.item() << "\n";
        }
    }

    NoGradGuard no_grad;
    std::cout << "\nпредсказания (ожидаем 0 1 1 0):\n" << model(x) << "\n";
}
