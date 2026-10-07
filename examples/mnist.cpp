// День 6: распознавание рукописных цифр MNIST.
//
//   mnist [папка_с_данными] [эпохи]
//
// Данные скачиваются скриптом scripts/download_mnist.ps1 (Windows)
// или scripts/download_mnist.sh (Linux/macOS) в папку data/.
#include <forge/forge.h>

#include <chrono>
#include <exception>
#include <iostream>
#include <string>

using namespace forge;

namespace {

// Точность на всём наборе, батчами, без построения графа.
float evaluate(nn::Module& model, const data::DataLoader& loader) {
    NoGradGuard no_grad;
    model.eval();
    float correct = 0.0f;
    float total = 0.0f;
    for (std::size_t b = 0; b < loader.num_batches(); ++b) {
        auto [x, y] = loader.batch(b);
        float n = static_cast<float>(x.shape()[0]);
        correct += nn::accuracy(model(x), y) * n;
        total += n;
    }
    model.train();
    return correct / total;
}

} // namespace

#ifndef FORGE_DATA_DIR
#define FORGE_DATA_DIR "data"
#endif

int run(int argc, char** argv) {
    std::string dir = argc > 1 ? argv[1] : FORGE_DATA_DIR;
    int epochs = argc > 2 ? std::stoi(argv[2]) : 5;

    std::cout << "Загружаю MNIST из " << dir << "/ ...\n";
    data::Dataset train = data::load_mnist(dir + "/train-images-idx3-ubyte",
                                           dir + "/train-labels-idx1-ubyte");
    data::Dataset test = data::load_mnist(dir + "/t10k-images-idx3-ubyte",
                                          dir + "/t10k-labels-idx1-ubyte");
    std::cout << "train: " << train.size() << " картинок, test: " << test.size() << "\n";

    manual_seed(0);
    nn::Sequential model;
    model.add<nn::Linear>(784, 128);
    model.add<nn::ReLU>();
    model.add<nn::Linear>(128, 10);
    std::cout << "параметров в модели: " << model.num_parameters() << "\n\n";

    optim::Adam optimizer(model.parameters(), 1e-3f);
    data::DataLoader train_loader(train, 64, /*shuffle=*/true);
    data::DataLoader test_loader(test, 1000, /*shuffle=*/false);

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        auto start = std::chrono::steady_clock::now();
        train_loader.reshuffle();
        float loss_sum = 0.0f;
        for (std::size_t b = 0; b < train_loader.num_batches(); ++b) {
            auto [x, y] = train_loader.batch(b);
            optimizer.zero_grad();
            Tensor loss = nn::cross_entropy(model(x), y);
            loss.backward();
            optimizer.step();
            loss_sum += loss.item();
        }
        double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        std::cout << "эпоха " << epoch << "  loss " << loss_sum / static_cast<float>(train_loader.num_batches())
                  << "  точность на test " << evaluate(model, test_loader) * 100.0f << "%"
                  << "  (" << seconds << " с)\n";
    }

    model.save("mnist_mlp.bin");
    std::cout << "\nвеса сохранены в mnist_mlp.bin\n";
    return 0;
}

int main(int argc, char** argv) {
    enable_utf8_console();
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "ошибка: " << e.what() << "\n";
        return 1;
    }
}
