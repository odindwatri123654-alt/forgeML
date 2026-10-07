#pragma once
#include <cstddef>
#include <memory>
#include <ostream>
#include <vector>

namespace forge {

using Shape = std::vector<std::size_t>;

// Плоский буфер чисел. Ничего не знает о форме.
class Storage {
public:
    explicit Storage(std::size_t size, float value = 0.0f);
    explicit Storage(std::vector<float> data);

    float* data();
    const float* data() const;
    std::size_t size() const;

private:
    std::vector<float> data_;
};

// "Тело" тензора: как интерпретировать Storage.
struct TensorImpl {
    std::shared_ptr<Storage> storage;
    Shape shape;
    Shape strides;            // в элементах, не в байтах
    std::size_t offset = 0;
    // День 3: здесь появятся grad, grad_fn, requires_grad
};

// Лёгкий handle. Копия Tensor = тот же тензор (как в PyTorch).
class Tensor {
public:
    Tensor() = default;

    // --- фабрики ---
    static Tensor zeros(const Shape& shape);
    static Tensor ones(const Shape& shape);
    static Tensor full(const Shape& shape, float value);
    static Tensor arange(float start, float end, float step = 1.0f);  // 1D
    static Tensor randn(const Shape& shape);                          // N(0, 1)
    static Tensor from_vector(std::vector<float> data, const Shape& shape);
    static Tensor eye(std::size_t n);                                 // единичная матрица

    // --- свойства ---
    const Shape& shape() const;
    const Shape& strides() const;
    std::size_t ndim() const;
    std::size_t numel() const;
    bool is_contiguous() const;

    // --- доступ: t.at({1, 2}) ---
    float& at(const Shape& index);
    float at(const Shape& index) const;

    // --- сырые данные (для contiguous-тензоров) ---
    float* data();
    const float* data() const;

    // --- views: новый взгляд на те же данные ---
    Tensor view(const Shape& shape) const;
    Tensor reshape(const Shape& shape) const;
    Tensor transpose(std::size_t dim0, std::size_t dim1) const;
    Tensor permute(const Shape& dims) const;
    Tensor expand(const Shape& shape) const;

    // --- копии ---
    Tensor contiguous() const;
    Tensor clone() const;

private:
    explicit Tensor(std::shared_ptr<TensorImpl> impl);
    std::shared_ptr<TensorImpl> impl_;
};

// --- утилиты ---
std::size_t numel(const Shape& shape);           // произведение размеров
Shape contiguous_strides(const Shape& shape);    // [2,3,4] -> [12,4,1]
void manual_seed(unsigned seed);                 // для воспроизводимого randn

std::ostream& operator<<(std::ostream& os, const Tensor& t);

} // namespace forge
