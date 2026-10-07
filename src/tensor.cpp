#include <forge/tensor.h>

#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>

namespace forge {

// ============================================================
// Storage
// ============================================================

Storage::Storage(std::size_t size, float value) : data_(size, value) {}

Storage::Storage(std::vector<float> data) : data_(std::move(data)) {}

float* Storage::data() { return data_.data(); }
const float* Storage::data() const { return data_.data(); }
std::size_t Storage::size() const { return data_.size(); }

// ============================================================
// Утилиты для Shape
// ============================================================

std::size_t numel(const Shape& shape) {
    std::size_t n = 1;
    for (std::size_t dim : shape) {
        n *= dim;
    }
    return n;
}

Shape contiguous_strides(const Shape& shape) {
    Shape strides(shape.size());
    std::size_t step = 1;
    for (std::size_t i = shape.size(); i-- > 0;) {
        strides[i] = step;
        step *= shape[i];
    }
    return strides;
}

// ============================================================
// Внутренние функции (видны только в этом файле)
// ============================================================

namespace {

std::mt19937& generator() {
    static std::mt19937 gen(42);
    return gen;
}

std::shared_ptr<TensorImpl> make_impl(std::shared_ptr<Storage> storage, const Shape& shape) {
    auto impl = std::make_shared<TensorImpl>();
    impl->storage = std::move(storage);
    impl->shape = shape;
    impl->strides = contiguous_strides(shape);
    impl->offset = 0;
    return impl;
}

std::size_t flat_index(const TensorImpl& impl, const Shape& index) {
    if (index.size() != impl.shape.size()) {
        throw std::invalid_argument("at(): expected " + std::to_string(impl.shape.size()) +
                                    " indices, got " + std::to_string(index.size()));
    }
    std::size_t pos = impl.offset;
    for (std::size_t d = 0; d < index.size(); ++d) {
        if (index[d] >= impl.shape[d]) {
            throw std::out_of_range("at(): index " + std::to_string(index[d]) +
                                    " is out of range for dim " + std::to_string(d) +
                                    " with size " + std::to_string(impl.shape[d]));
        }
        pos += index[d] * impl.strides[d];
    }
    return pos;
}

} // namespace

void manual_seed(unsigned seed) { generator().seed(seed); }

// ============================================================
// Tensor: конструктор и фабрики
// ============================================================

Tensor::Tensor(std::shared_ptr<TensorImpl> impl) : impl_(std::move(impl)) {}

Tensor Tensor::full(const Shape& shape, float value) {
    auto storage = std::make_shared<Storage>(forge::numel(shape), value);
    return Tensor(make_impl(std::move(storage), shape));
}

Tensor Tensor::zeros(const Shape& shape) { return full(shape, 0.0f); }
Tensor Tensor::ones(const Shape& shape) { return full(shape, 1.0f); }

Tensor Tensor::from_vector(std::vector<float> data, const Shape& shape) {
    if (data.size() != forge::numel(shape)) {
        throw std::invalid_argument("from_vector(): data has " + std::to_string(data.size()) +
                                    " elements, but shape needs " +
                                    std::to_string(forge::numel(shape)));
    }
    auto storage = std::make_shared<Storage>(std::move(data));
    return Tensor(make_impl(std::move(storage), shape));
}

Tensor Tensor::arange(float start, float end, float step) {
    if (step == 0.0f) {
        throw std::invalid_argument("arange(): step must be non-zero");
    }
    float count = std::ceil((end - start) / step);
    std::size_t n = count > 0.0f ? static_cast<std::size_t>(count) : 0;

    std::vector<float> data(n);
    for (std::size_t i = 0; i < n; ++i) {
        data[i] = start + static_cast<float>(i) * step;
    }
    return from_vector(std::move(data), {n});
}

Tensor Tensor::randn(const Shape& shape) {
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> data(forge::numel(shape));
    for (float& x : data) {
        x = dist(generator());
    }
    return from_vector(std::move(data), shape);
}

Tensor Tensor::eye(std::size_t n) {
    Tensor t = Tensor::zeros({n, n});
    for (std::size_t i = 0; i < n; ++i) {
        t.at({i, i}) = 1.0f;
    }
    return t;
}

// ============================================================
// Tensor: свойства
// ============================================================

const Shape& Tensor::shape() const { return impl_->shape; }
const Shape& Tensor::strides() const { return impl_->strides; }
std::size_t Tensor::ndim() const { return impl_->shape.size(); }
std::size_t Tensor::numel() const { return forge::numel(impl_->shape); }

bool Tensor::is_contiguous() const {
    Shape expected = contiguous_strides(impl_->shape);
    for (std::size_t d = 0; d < ndim(); ++d) {
        if (impl_->shape[d] != 1 && impl_->strides[d] != expected[d]) {
            return false;
        }
    }
    return true;
}

// ============================================================
// Tensor: доступ к элементам
// ============================================================

float& Tensor::at(const Shape& index) {
    return impl_->storage->data()[flat_index(*impl_, index)];
}

float Tensor::at(const Shape& index) const {
    return impl_->storage->data()[flat_index(*impl_, index)];
}

// ============================================================
// Печать
// ============================================================

namespace {

void print_recursive(std::ostream& os, const Tensor& t, Shape& index, std::size_t dim) {
    if (dim == t.ndim()) {
        os << t.at(index);
        return;
    }
    os << '[';
    std::size_t size = t.shape()[dim];
    for (std::size_t i = 0; i < size; ++i) {
        index[dim] = i;
        print_recursive(os, t, index, dim + 1);
        if (i + 1 < size) {
            os << ',';
            if (dim + 1 == t.ndim()) {
                os << ' ';
            } else {
                os << '\n' << std::string(7 + dim + 1, ' ');
            }
        }
    }
    os << ']';
}

} // namespace

std::ostream& operator<<(std::ostream& os, const Tensor& t) {
    Shape index(t.ndim(), 0);
    os << "tensor(";
    print_recursive(os, t, index, 0);
    os << ')';
    return os;
}

// ============================================================
// День 2: сырые данные
// ============================================================

float* Tensor::data() { return impl_->storage->data() + impl_->offset; }
const float* Tensor::data() const { return impl_->storage->data() + impl_->offset; }

// ============================================================
// День 2: views
// ============================================================

namespace {

std::shared_ptr<TensorImpl> make_view(const TensorImpl& base, Shape shape, Shape strides) {
    auto impl = std::make_shared<TensorImpl>();
    impl->storage = base.storage;
    impl->shape = std::move(shape);
    impl->strides = std::move(strides);
    impl->offset = base.offset;
    return impl;
}

std::string shape_to_string(const Shape& shape) {
    std::string s = "[";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) s += ", ";
        s += std::to_string(shape[i]);
    }
    return s + "]";
}

} // namespace

Tensor Tensor::view(const Shape& shape) const {
    if (!is_contiguous()) {
        throw std::invalid_argument("view(): tensor is not contiguous, use reshape()");
    }
    if (forge::numel(shape) != numel()) {
        throw std::invalid_argument("view(): cannot view " + shape_to_string(impl_->shape) +
                                    " as " + shape_to_string(shape));
    }
    return Tensor(make_view(*impl_, shape, contiguous_strides(shape)));
}

Tensor Tensor::reshape(const Shape& shape) const {
    if (is_contiguous()) {
        return view(shape);
    }
    return contiguous().view(shape);
}

Tensor Tensor::transpose(std::size_t dim0, std::size_t dim1) const {
    if (dim0 >= ndim() || dim1 >= ndim()) {
        throw std::out_of_range("transpose(): dim out of range for tensor with " +
                                std::to_string(ndim()) + " dims");
    }
    Shape shape = impl_->shape;
    Shape strides = impl_->strides;
    std::swap(shape[dim0], shape[dim1]);
    std::swap(strides[dim0], strides[dim1]);
    return Tensor(make_view(*impl_, std::move(shape), std::move(strides)));
}

Tensor Tensor::permute(const Shape& dims) const {
    if (dims.size() != ndim()) {
        throw std::invalid_argument("permute(): expected " + std::to_string(ndim()) + " dims");
    }
    Shape shape(ndim());
    Shape strides(ndim());
    std::vector<bool> used(ndim(), false);
    for (std::size_t i = 0; i < dims.size(); ++i) {
        std::size_t d = dims[i];
        if (d >= ndim() || used[d]) {
            throw std::invalid_argument("permute(): dims must be a permutation of 0.." +
                                        std::to_string(ndim() - 1));
        }
        used[d] = true;
        shape[i] = impl_->shape[d];
        strides[i] = impl_->strides[d];
    }
    return Tensor(make_view(*impl_, std::move(shape), std::move(strides)));
}

Tensor Tensor::expand(const Shape& shape) const {
    if (shape.size() < ndim()) {
        throw std::invalid_argument("expand(): cannot expand " + shape_to_string(impl_->shape) +
                                    " to " + shape_to_string(shape));
    }
    std::size_t extra = shape.size() - ndim();
    Shape strides(shape.size(), 0);
    for (std::size_t i = 0; i < ndim(); ++i) {
        std::size_t src = impl_->shape[i];
        std::size_t dst = shape[extra + i];
        if (src == dst) {
            strides[extra + i] = impl_->strides[i];
        } else if (src == 1) {
            strides[extra + i] = 0;
        } else {
            throw std::invalid_argument("expand(): cannot expand " +
                                        shape_to_string(impl_->shape) + " to " +
                                        shape_to_string(shape));
        }
    }
    return Tensor(make_view(*impl_, shape, std::move(strides)));
}

// ============================================================
// День 2: копии
// ============================================================

Tensor Tensor::clone() const {
    const Shape& shape = impl_->shape;
    const Shape& strides = impl_->strides;
    const float* base = impl_->storage->data();

    std::vector<float> out(numel());
    Shape index(ndim(), 0);
    for (std::size_t i = 0; i < out.size(); ++i) {
        std::size_t pos = impl_->offset;
        for (std::size_t d = 0; d < ndim(); ++d) {
            pos += index[d] * strides[d];
        }
        out[i] = base[pos];

        for (std::size_t d = ndim(); d-- > 0;) {
            if (++index[d] < shape[d]) {
                break;
            }
            index[d] = 0;
        }
    }
    return from_vector(std::move(out), shape);
}

Tensor Tensor::contiguous() const {
    if (is_contiguous()) {
        return *this;
    }
    return clone();
}

} // namespace forge
