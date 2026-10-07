#include <forge/data.h>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <numeric>
#include <stdexcept>

namespace forge::data {

// ============================================================
// MNIST (формат IDX)
// ============================================================

namespace {

// В IDX числа записаны в big-endian: старший байт первым.
// Собираем uint32 вручную, чтобы не зависеть от порядка байт процессора.
std::uint32_t read_be_u32(std::ifstream& in) {
    unsigned char b[4] = {};
    in.read(reinterpret_cast<char*>(b), 4);
    if (!in) {
        throw std::runtime_error("load_mnist(): unexpected end of file");
    }
    return (std::uint32_t(b[0]) << 24) | (std::uint32_t(b[1]) << 16) |
           (std::uint32_t(b[2]) << 8) | std::uint32_t(b[3]);
}

std::ifstream open_idx(const std::string& path, std::uint32_t expected_magic) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("load_mnist(): cannot open " + path +
                                 " (run scripts/download_mnist first)");
    }
    if (read_be_u32(in) != expected_magic) {
        throw std::runtime_error("load_mnist(): " + path + " is not an IDX file of expected type");
    }
    return in;
}

std::vector<unsigned char> read_bytes(std::ifstream& in, std::size_t count) {
    std::vector<unsigned char> bytes(count);
    in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(count));
    if (!in) {
        throw std::runtime_error("load_mnist(): file is truncated");
    }
    return bytes;
}

} // namespace

Dataset load_mnist(const std::string& images_path, const std::string& labels_path) {
    std::ifstream images = open_idx(images_path, 2051);  // 0x00000803
    std::size_t n = read_be_u32(images);
    std::size_t rows = read_be_u32(images);
    std::size_t cols = read_be_u32(images);

    std::ifstream labels = open_idx(labels_path, 2049);  // 0x00000801
    if (read_be_u32(labels) != n) {
        throw std::runtime_error("load_mnist(): images and labels count differ");
    }

    std::vector<unsigned char> pixels = read_bytes(images, n * rows * cols);
    std::vector<unsigned char> digits = read_bytes(labels, n);

    std::vector<float> x(pixels.size());
    for (std::size_t i = 0; i < pixels.size(); ++i) {
        x[i] = static_cast<float>(pixels[i]) / 255.0f;  // 0..255 -> 0..1
    }
    std::vector<float> y(digits.begin(), digits.end());

    return Dataset{Tensor::from_vector(std::move(x), {n, rows * cols}),
                   Tensor::from_vector(std::move(y), {n})};
}

// ============================================================
// DataLoader
// ============================================================

DataLoader::DataLoader(Dataset dataset, std::size_t batch_size, bool shuffle, unsigned seed)
    : dataset_(std::move(dataset)), batch_size_(batch_size), shuffle_(shuffle), rng_(seed) {
    if (batch_size_ == 0) {
        throw std::invalid_argument("DataLoader: batch_size must be > 0");
    }
    if (dataset_.targets.shape()[0] != dataset_.size()) {
        throw std::invalid_argument("DataLoader: inputs and targets have different lengths");
    }
    // копии contiguous — чтобы копировать строки простым циклом
    dataset_.inputs = dataset_.inputs.detach().contiguous();
    dataset_.targets = dataset_.targets.detach().contiguous();
    order_.resize(dataset_.size());
    std::iota(order_.begin(), order_.end(), 0);  // 0, 1, 2, ..., N-1
    reshuffle();
}

std::size_t DataLoader::num_batches() const {
    return (dataset_.size() + batch_size_ - 1) / batch_size_;  // округление вверх
}

void DataLoader::reshuffle() {
    if (shuffle_) {
        std::shuffle(order_.begin(), order_.end(), rng_);
    }
}

namespace {

// Собирает строки rows[begin..end) тензора src [N, ...] в новый тензор [count, ...].
Tensor gather_rows(const Tensor& src, const std::vector<std::size_t>& order, std::size_t begin,
                   std::size_t end) {
    Shape shape = src.shape();
    std::size_t row_size = src.numel() / shape[0];
    shape[0] = end - begin;
    std::vector<float> out(shape[0] * row_size);
    const float* data = src.data();
    for (std::size_t i = begin; i < end; ++i) {
        const float* row = data + order[i] * row_size;
        std::copy(row, row + row_size, out.begin() + static_cast<std::ptrdiff_t>((i - begin) * row_size));
    }
    return Tensor::from_vector(std::move(out), shape);
}

} // namespace

std::pair<Tensor, Tensor> DataLoader::batch(std::size_t index) const {
    if (index >= num_batches()) {
        throw std::out_of_range("DataLoader::batch(): index out of range");
    }
    std::size_t begin = index * batch_size_;
    std::size_t end = std::min(begin + batch_size_, dataset_.size());
    return {gather_rows(dataset_.inputs, order_, begin, end),
            gather_rows(dataset_.targets, order_, begin, end)};
}

} // namespace forge::data
