#pragma once
#include <forge/tensor.h>

#include <random>
#include <string>
#include <utility>
#include <vector>

namespace forge::data {

// Набор примеров: inputs[i] — вход, targets[i] — правильный ответ.
struct Dataset {
    Tensor inputs;   // [N, ...]
    Tensor targets;  // [N] или [N, ...]
    std::size_t size() const { return inputs.shape()[0]; }
};

// Читает MNIST из распакованных файлов формата IDX.
// inputs: [N, 784], пиксели в [0, 1];  targets: [N], цифры 0..9.
Dataset load_mnist(const std::string& images_path, const std::string& labels_path);

// Делит Dataset на мини-батчи и перемешивает их каждую эпоху.
class DataLoader {
public:
    DataLoader(Dataset dataset, std::size_t batch_size, bool shuffle = true, unsigned seed = 0);

    std::size_t num_batches() const;
    std::pair<Tensor, Tensor> batch(std::size_t index) const;  // (inputs, targets)
    void reshuffle();  // новый случайный порядок (вызывать в начале эпохи)

private:
    Dataset dataset_;
    std::size_t batch_size_;
    bool shuffle_;
    std::vector<std::size_t> order_;
    std::mt19937 rng_;
};

} // namespace forge::data
