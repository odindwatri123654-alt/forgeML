#include <forge/ops.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace forge {

// ============================================================
// Broadcasting
// ============================================================

Shape broadcast_shapes(const Shape& a, const Shape& b) {
    std::size_t ndim = std::max(a.size(), b.size());
    Shape result(ndim);
    for (std::size_t i = 0; i < ndim; ++i) {
        // идём с конца: i = 0 — последняя ось
        std::size_t da = i < a.size() ? a[a.size() - 1 - i] : 1;
        std::size_t db = i < b.size() ? b[b.size() - 1 - i] : 1;
        if (da != db && da != 1 && db != 1) {
            throw std::invalid_argument("broadcast_shapes(): sizes " + std::to_string(da) +
                                        " and " + std::to_string(db) + " are incompatible");
        }
        result[ndim - 1 - i] = std::max(da, db);
    }
    return result;
}

// ============================================================
// Общие шаблоны для поэлементных операций
// ============================================================

namespace {

template <typename F>
Tensor unary_op(const Tensor& a, F f) {
    Tensor src = a.contiguous();
    Tensor out = Tensor::zeros(src.shape());
    const float* x = src.data();
    float* y = out.data();
    for (std::size_t i = 0; i < out.numel(); ++i) {
        y[i] = f(x[i]);
    }
    return out;
}

template <typename F>
Tensor binary_op(const Tensor& a, const Tensor& b, F f) {
    Shape shape = broadcast_shapes(a.shape(), b.shape());
    Tensor ac = a.expand(shape).contiguous();
    Tensor bc = b.expand(shape).contiguous();
    Tensor out = Tensor::zeros(shape);
    const float* x = ac.data();
    const float* y = bc.data();
    float* z = out.data();
    for (std::size_t i = 0; i < out.numel(); ++i) {
        z[i] = f(x[i], y[i]);
    }
    return out;
}

Tensor scalar(float value) { return Tensor::full({}, value); }

} // namespace

// ============================================================
// Поэлементные операции
// ============================================================

Tensor add(const Tensor& a, const Tensor& b) {
    return binary_op(a, b, [](float x, float y) { return x + y; });
}
Tensor sub(const Tensor& a, const Tensor& b) {
    return binary_op(a, b, [](float x, float y) { return x - y; });
}
Tensor mul(const Tensor& a, const Tensor& b) {
    return binary_op(a, b, [](float x, float y) { return x * y; });
}
Tensor div(const Tensor& a, const Tensor& b) {
    return binary_op(a, b, [](float x, float y) { return x / y; });
}

Tensor neg(const Tensor& a) {
    return unary_op(a, [](float x) { return -x; });
}
Tensor exp(const Tensor& a) {
    return unary_op(a, [](float x) { return std::exp(x); });
}
Tensor log(const Tensor& a) {
    return unary_op(a, [](float x) { return std::log(x); });
}
Tensor relu(const Tensor& a) {
    return unary_op(a, [](float x) { return x > 0.0f ? x : 0.0f; });
}
Tensor pow(const Tensor& a, float exponent) {
    return unary_op(a, [exponent](float x) { return std::pow(x, exponent); });
}

// ============================================================
// Редукции
// ============================================================

namespace {

// Общая схема: смотрим на тензор как на [outer, size, inner],
// где size — ось, по которой сворачиваем.
template <typename Init, typename Combine>
Tensor reduce_dim(const Tensor& a, std::size_t dim, bool keepdim, Init init, Combine combine) {
    if (dim >= a.ndim()) {
        throw std::out_of_range("reduce: dim " + std::to_string(dim) +
                                " out of range for tensor with " + std::to_string(a.ndim()) +
                                " dims");
    }
    Tensor src = a.contiguous();
    const Shape& shape = src.shape();

    std::size_t outer = 1;
    for (std::size_t d = 0; d < dim; ++d) outer *= shape[d];
    std::size_t size = shape[dim];
    std::size_t inner = 1;
    for (std::size_t d = dim + 1; d < shape.size(); ++d) inner *= shape[d];

    Shape out_shape = shape;
    if (keepdim) {
        out_shape[dim] = 1;
    } else {
        out_shape.erase(out_shape.begin() + static_cast<std::ptrdiff_t>(dim));
    }

    Tensor out = Tensor::zeros(out_shape);
    const float* x = src.data();
    float* y = out.data();
    for (std::size_t o = 0; o < outer; ++o) {
        for (std::size_t i = 0; i < inner; ++i) {
            float acc = init;
            for (std::size_t k = 0; k < size; ++k) {
                acc = combine(acc, x[(o * size + k) * inner + i]);
            }
            y[o * inner + i] = acc;
        }
    }
    return out;
}

} // namespace

Tensor sum(const Tensor& a) {
    Tensor src = a.contiguous();
    const float* x = src.data();
    double acc = 0.0;
    for (std::size_t i = 0; i < src.numel(); ++i) {
        acc += x[i];
    }
    return scalar(static_cast<float>(acc));
}

Tensor sum(const Tensor& a, std::size_t dim, bool keepdim) {
    return reduce_dim(a, dim, keepdim, 0.0f, [](float acc, float x) { return acc + x; });
}

Tensor mean(const Tensor& a) {
    return sum(a) / static_cast<float>(a.numel());
}

Tensor mean(const Tensor& a, std::size_t dim, bool keepdim) {
    Tensor s = sum(a, dim, keepdim);  // сначала sum: он проверит, что dim корректен
    return s / static_cast<float>(a.shape()[dim]);
}

Tensor max(const Tensor& a, std::size_t dim, bool keepdim) {
    return reduce_dim(a, dim, keepdim, -std::numeric_limits<float>::infinity(),
                      [](float acc, float x) { return x > acc ? x : acc; });
}

// ============================================================
// Матричное умножение
// ============================================================

Tensor matmul(const Tensor& a, const Tensor& b) {
    if (a.ndim() != 2 || b.ndim() != 2) {
        throw std::invalid_argument("matmul(): only 2D tensors are supported");
    }
    std::size_t n = a.shape()[0];
    std::size_t k = a.shape()[1];
    std::size_t m = b.shape()[1];
    if (b.shape()[0] != k) {
        throw std::invalid_argument("matmul(): shapes [" + std::to_string(n) + ", " +
                                    std::to_string(k) + "] and [" +
                                    std::to_string(b.shape()[0]) + ", " + std::to_string(m) +
                                    "] are incompatible");
    }
    Tensor ac = a.contiguous();
    Tensor bc = b.contiguous();
    Tensor out = Tensor::zeros({n, m});
    const float* A = ac.data();
    const float* B = bc.data();
    float* C = out.data();

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t p = 0; p < k; ++p) {
            float a_ip = A[i * k + p];
            for (std::size_t j = 0; j < m; ++j) {
                C[i * m + j] += a_ip * B[p * m + j];
            }
        }
    }
    return out;
}

// ============================================================
// Операторы
// ============================================================

Tensor operator+(const Tensor& a, const Tensor& b) { return add(a, b); }
Tensor operator-(const Tensor& a, const Tensor& b) { return sub(a, b); }
Tensor operator*(const Tensor& a, const Tensor& b) { return mul(a, b); }
Tensor operator/(const Tensor& a, const Tensor& b) { return div(a, b); }
Tensor operator-(const Tensor& a) { return neg(a); }

Tensor operator+(const Tensor& a, float b) { return add(a, scalar(b)); }
Tensor operator-(const Tensor& a, float b) { return sub(a, scalar(b)); }
Tensor operator*(const Tensor& a, float b) { return mul(a, scalar(b)); }
Tensor operator/(const Tensor& a, float b) { return div(a, scalar(b)); }
Tensor operator+(float a, const Tensor& b) { return add(scalar(a), b); }
Tensor operator-(float a, const Tensor& b) { return sub(scalar(a), b); }
Tensor operator*(float a, const Tensor& b) { return mul(scalar(a), b); }
Tensor operator/(float a, const Tensor& b) { return div(scalar(a), b); }

} // namespace forge
