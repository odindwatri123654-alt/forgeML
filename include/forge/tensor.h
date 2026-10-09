#pragma once
#include <cstddef>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

namespace forge {

using Shape = std::vector<std::size_t>;

struct Node;  // узел графа autograd (autograd.h)

// Плоский буфер чисел. Ничего не знает о форме.
class Storage {
public:
    explicit Storage(std::size_t size, float value = 0.0f);
    explicit Storage(std::vector<float> data);

    float* data();
    const float* data() const;
    std::size_t size() const;

    // Счётчик изменений: растёт при каждом доступе на запись (неконстантные
    // Tensor::data() и Tensor::at()). По нему autograd узнаёт, что данные,
    // использованные в forward, поменяли до backward.
    std::size_t version() const { return version_; }
    void bump_version() { ++version_; }

private:
    std::vector<float> data_;
    std::size_t version_ = 0;
};

// "Тело" тензора: как интерпретировать Storage.
struct TensorImpl {
    std::shared_ptr<Storage> storage;
    Shape shape;
    Shape strides;            // в элементах, не в байтах
    std::size_t offset = 0;

    // --- autograd (День 3) ---
    bool requires_grad = false;
    std::shared_ptr<TensorImpl> grad;   // накопленный градиент (только у листьев)
    std::shared_ptr<Node> grad_fn;      // операция, которая создала этот тензор
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
    static Tensor rand(const Shape& shape);                           // U[0, 1)
    static Tensor from_vector(std::vector<float> data, const Shape& shape);
    static Tensor eye(std::size_t n);                                 // единичная матрица

    // --- свойства ---
    bool defined() const;                // false у Tensor(), созданного по умолчанию
    const Shape& shape() const;
    const Shape& strides() const;
    std::size_t ndim() const;
    std::size_t numel() const;
    bool is_contiguous() const;

    // --- доступ: t.at({1, 2}) ---
    // Неконстантные at() и data() считаются записью (см. Storage::version):
    // чтобы только прочитать, вызывайте их у const Tensor или используйте item().
    float& at(const Shape& index);
    float at(const Shape& index) const;
    float item() const;                  // значение тензора из одного элемента

    // --- сырые данные (для contiguous-тензоров) ---
    float* data();
    const float* data() const;

    // --- views: новый взгляд на те же данные ---
    Tensor view(const Shape& shape) const;
    Tensor reshape(const Shape& shape) const;
    Tensor transpose(std::size_t dim0, std::size_t dim1) const;
    Tensor permute(const Shape& dims) const;
    Tensor expand(const Shape& shape) const;
    Tensor unsqueeze(std::size_t dim) const;   // вставить ось размера 1
    Tensor squeeze(std::size_t dim) const;     // убрать ось размера 1

    // --- копии ---
    Tensor contiguous() const;
    Tensor clone() const;

    // --- autograd (День 3) ---
    bool requires_grad() const;
    Tensor requires_grad_(bool value = true);  // включить отслеживание (только у листьев)
    bool is_leaf() const;                      // создан пользователем, а не операцией
    Tensor grad() const;                       // накопленный градиент (или пустой Tensor)
    void zero_grad();                          // забыть накопленный градиент
    Tensor detach() const;                     // те же данные, но вне графа
    void backward() const;                     // для скаляра: d(this)/d(листья)
    void backward(const Tensor& grad) const;   // для тензора: с явным градиентом выхода

    // Внутреннее: доступ к TensorImpl для autograd и библиотечного кода.
    const std::shared_ptr<TensorImpl>& impl() const { return impl_; }

private:
    explicit Tensor(std::shared_ptr<TensorImpl> impl);
    TensorImpl* get() const;   // impl_ с проверкой "тензор определён"

    std::shared_ptr<TensorImpl> impl_;
};

// --- утилиты ---
std::size_t numel(const Shape& shape);           // произведение размеров
Shape contiguous_strides(const Shape& shape);    // [2,3,4] -> [12,4,1]
void manual_seed(unsigned seed);                 // для воспроизводимого randn / rand
std::string shape_to_string(const Shape& shape); // {2, 3} -> "[2, 3]"

std::ostream& operator<<(std::ostream& os, const Tensor& t);

} // namespace forge
