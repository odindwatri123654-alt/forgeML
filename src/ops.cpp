#include <forge/autograd.h>
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
                                        " and " + std::to_string(db) + " are incompatible (" +
                                        shape_to_string(a) + " vs " + shape_to_string(b) + ")");
        }
        // ось размера 1 растягивается до размера другой — в том числе до 0
        result[ndim - 1 - i] = (da == 1) ? db : da;
    }
    return result;
}

// ============================================================
// Общие шаблоны: считают ТОЛЬКО значения (без графа)
// ============================================================

namespace {

template <typename F>
Tensor unary_op(const Tensor& a, F f) {
    const Tensor src = a.detach().contiguous();
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
    const Tensor ac = a.detach().expand(shape).contiguous();
    const Tensor bc = b.detach().expand(shape).contiguous();
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

// Градиент нужен только тем входам, у которых requires_grad.
Tensor grad_if(const Tensor& input, const Tensor& grad) {
    return input.requires_grad() ? sum_to_shape(grad, input.shape()) : Tensor();
}

} // namespace

// ============================================================
// Поэлементные операции с двумя тензорами
// ============================================================

Tensor add(const Tensor& a, const Tensor& b) {
    Tensor out = binary_op(a, b, [](float x, float y) { return x + y; });
    // d(a+b)/da = 1, d(a+b)/db = 1
    return record_op(out, {a, b}, "AddBackward", [a, b](const Tensor& g) {
        return std::vector<Tensor>{grad_if(a, g), grad_if(b, g)};
    });
}

Tensor sub(const Tensor& a, const Tensor& b) {
    Tensor out = binary_op(a, b, [](float x, float y) { return x - y; });
    // d(a-b)/da = 1, d(a-b)/db = -1
    return record_op(out, {a, b}, "SubBackward", [a, b](const Tensor& g) {
        return std::vector<Tensor>{grad_if(a, g), grad_if(b, -g)};
    });
}

Tensor mul(const Tensor& a, const Tensor& b) {
    Tensor out = binary_op(a, b, [](float x, float y) { return x * y; });
    // d(a*b)/da = b, d(a*b)/db = a
    return record_op(out, {a, b}, "MulBackward", [a, b](const Tensor& g) {
        Tensor da = a.requires_grad() ? sum_to_shape(g * b, a.shape()) : Tensor();
        Tensor db = b.requires_grad() ? sum_to_shape(g * a, b.shape()) : Tensor();
        return std::vector<Tensor>{da, db};
    });
}

Tensor div(const Tensor& a, const Tensor& b) {
    Tensor out = binary_op(a, b, [](float x, float y) { return x / y; });
    // d(a/b)/da = 1/b, d(a/b)/db = -a/b^2
    return record_op(out, {a, b}, "DivBackward", [a, b](const Tensor& g) {
        Tensor da = a.requires_grad() ? sum_to_shape(g / b, a.shape()) : Tensor();
        Tensor db = b.requires_grad() ? sum_to_shape(-g * a / (b * b), b.shape()) : Tensor();
        return std::vector<Tensor>{da, db};
    });
}

// ============================================================
// Поэлементные операции с одним тензором
// ============================================================

Tensor neg(const Tensor& a) {
    Tensor out = unary_op(a, [](float x) { return -x; });
    return record_op(out, {a}, "NegBackward", [](const Tensor& g) {
        return std::vector<Tensor>{-g};
    });
}

Tensor exp(const Tensor& a) {
    Tensor out = unary_op(a, [](float x) { return std::exp(x); });
    // (e^x)' = e^x
    return record_op(out, {a}, "ExpBackward", [a](const Tensor& g) {
        return std::vector<Tensor>{g * exp(a)};
    });
}

Tensor log(const Tensor& a) {
    Tensor out = unary_op(a, [](float x) { return std::log(x); });
    // (ln x)' = 1/x
    return record_op(out, {a}, "LogBackward", [a](const Tensor& g) {
        return std::vector<Tensor>{g / a};
    });
}

Tensor relu(const Tensor& a) {
    Tensor out = unary_op(a, [](float x) { return x > 0.0f ? x : 0.0f; });
    // relu'(x) = 1 при x > 0, иначе 0
    return record_op(out, {a}, "ReluBackward", [a](const Tensor& g) {
        Tensor mask = unary_op(a, [](float x) { return x > 0.0f ? 1.0f : 0.0f; });
        return std::vector<Tensor>{g * mask};
    });
}

Tensor sigmoid(const Tensor& a) {
    Tensor out = unary_op(a, [](float x) { return 1.0f / (1.0f + std::exp(-x)); });
    // s'(x) = s(x) * (1 - s(x))
    return record_op(out, {a}, "SigmoidBackward", [a](const Tensor& g) {
        Tensor s = sigmoid(a);
        return std::vector<Tensor>{g * s * (1.0f - s)};
    });
}

Tensor tanh(const Tensor& a) {
    Tensor out = unary_op(a, [](float x) { return std::tanh(x); });
    // tanh'(x) = 1 - tanh(x)^2
    return record_op(out, {a}, "TanhBackward", [a](const Tensor& g) {
        Tensor t = tanh(a);
        return std::vector<Tensor>{g * (1.0f - t * t)};
    });
}

Tensor pow(const Tensor& a, float exponent) {
    Tensor out = unary_op(a, [exponent](float x) { return std::pow(x, exponent); });
    // (x^p)' = p * x^(p-1)
    return record_op(out, {a}, "PowBackward", [a, exponent](const Tensor& g) {
        return std::vector<Tensor>{g * exponent * pow(a, exponent - 1.0f)};
    });
}

// ============================================================
// Редукции
// ============================================================

namespace {

struct DimSplit {
    std::size_t outer = 1;  // произведение осей левее dim
    std::size_t size = 1;   // размер самой оси dim
    std::size_t inner = 1;  // произведение осей правее dim
};

DimSplit split_at(const Shape& shape, std::size_t dim) {
    if (dim >= shape.size()) {
        throw std::out_of_range("reduce: dim " + std::to_string(dim) +
                                " out of range for tensor with " + std::to_string(shape.size()) +
                                " dims");
    }
    DimSplit s;
    for (std::size_t d = 0; d < dim; ++d) s.outer *= shape[d];
    s.size = shape[dim];
    for (std::size_t d = dim + 1; d < shape.size(); ++d) s.inner *= shape[d];
    return s;
}

Shape reduced_shape(Shape shape, std::size_t dim, bool keepdim) {
    if (keepdim) {
        shape[dim] = 1;
    } else {
        shape.erase(shape.begin() + static_cast<std::ptrdiff_t>(dim));
    }
    return shape;
}

// Общая схема: смотрим на тензор как на [outer, size, inner]
// и сворачиваем среднюю ось. Тип аккумулятора задаёт init:
// для суммы это double (меньше ошибка округления на длинных осях).
template <typename Acc, typename Combine>
Tensor reduce_dim(const Tensor& a, std::size_t dim, bool keepdim, Acc init, Combine combine) {
    DimSplit s = split_at(a.shape(), dim);
    const Tensor src = a.detach().contiguous();
    Tensor out = Tensor::zeros(reduced_shape(a.shape(), dim, keepdim));
    const float* x = src.data();
    float* y = out.data();
    for (std::size_t o = 0; o < s.outer; ++o) {
        for (std::size_t i = 0; i < s.inner; ++i) {
            Acc acc = init;
            for (std::size_t k = 0; k < s.size; ++k) {
                acc = combine(acc, x[(o * s.size + k) * s.inner + i]);
            }
            y[o * s.inner + i] = static_cast<float>(acc);
        }
    }
    return out;
}

} // namespace

Tensor sum(const Tensor& a) {
    const Tensor src = a.detach().contiguous();
    const float* x = src.data();
    double acc = 0.0;  // double: меньше ошибка округления при длинной сумме
    for (std::size_t i = 0; i < src.numel(); ++i) {
        acc += x[i];
    }
    Tensor out = scalar(static_cast<float>(acc));
    // каждый элемент входит в сумму с коэффициентом 1
    return record_op(out, {a}, "SumBackward", [shape = a.shape()](const Tensor& g) {
        return std::vector<Tensor>{g.expand(shape)};
    });
}

Tensor sum(const Tensor& a, std::size_t dim, bool keepdim) {
    Tensor out = reduce_dim(a, dim, keepdim, 0.0, [](double acc, float x) { return acc + x; });
    return record_op(out, {a}, "SumDimBackward",
                     [shape = a.shape(), dim, keepdim](const Tensor& g) {
                         Tensor gk = keepdim ? g : g.unsqueeze(dim);
                         return std::vector<Tensor>{gk.expand(shape)};
                     });
}

// mean собран из sum и деления — его backward autograd выведет сам.
Tensor mean(const Tensor& a) {
    return sum(a) / static_cast<float>(a.numel());
}

Tensor mean(const Tensor& a, std::size_t dim, bool keepdim) {
    Tensor s = sum(a, dim, keepdim);  // сначала sum: он проверит, что dim корректен
    return s / static_cast<float>(a.shape()[dim]);
}

Tensor max(const Tensor& a, std::size_t dim, bool keepdim) {
    // NaN "заразен", как в PyTorch: если в строке есть NaN, максимум — NaN.
    // Иначе испорченное обучение выглядело бы нормальным.
    Tensor out = reduce_dim(a, dim, keepdim, -std::numeric_limits<float>::infinity(),
                            [](float acc, float x) {
                                return (x > acc || std::isnan(x)) && !std::isnan(acc) ? x : acc;
                            });
    // градиент идёт только в элемент, который оказался максимумом
    return record_op(out, {a}, "MaxBackward", [a, dim, keepdim](const Tensor& g) {
        Tensor m = max(a, dim, /*keepdim=*/true);
        Tensor mask = binary_op(a, m, [](float x, float y) { return x == y ? 1.0f : 0.0f; });
        Tensor gk = keepdim ? g : g.unsqueeze(dim);
        return std::vector<Tensor>{mask * gk};
    });
}

Tensor argmax(const Tensor& a, std::size_t dim) {
    DimSplit s = split_at(a.shape(), dim);
    const Tensor src = a.detach().contiguous();
    Tensor out = Tensor::zeros(reduced_shape(a.shape(), dim, /*keepdim=*/false));
    const float* x = src.data();
    float* y = out.data();
    for (std::size_t o = 0; o < s.outer; ++o) {
        for (std::size_t i = 0; i < s.inner; ++i) {
            std::size_t best = 0;
            float best_value = -std::numeric_limits<float>::infinity();
            for (std::size_t k = 0; k < s.size; ++k) {
                float v = x[(o * s.size + k) * s.inner + i];
                // NaN считается максимумом (как в PyTorch) — индекс первого NaN
                if (std::isnan(best_value)) break;
                if (v > best_value || std::isnan(v)) {
                    best_value = v;
                    best = k;
                }
            }
            y[o * s.inner + i] = static_cast<float>(best);
        }
    }
    return out;  // индексы не дифференцируемы — граф не нужен
}

// ============================================================
// Softmax (собраны из уже дифференцируемых операций)
// ============================================================

Tensor log_softmax(const Tensor& a, std::size_t dim) {
    // log_softmax(x) = x - m - log(sum(exp(x - m))), m = max(x).
    // Вычитание m не меняет результат, но спасает exp от переполнения.
    // m отсоединён от графа: на градиент он не влияет.
    Tensor shifted = a - max(a, dim, /*keepdim=*/true).detach();
    return shifted - log(sum(exp(shifted), dim, /*keepdim=*/true));
}

Tensor softmax(const Tensor& a, std::size_t dim) {
    Tensor e = exp(a - max(a, dim, /*keepdim=*/true).detach());
    return e / sum(e, dim, /*keepdim=*/true);
}

// ============================================================
// Матричное умножение
// ============================================================

namespace {

// Только числа: C = A · B для contiguous матриц.
Tensor matmul_values(const Tensor& a, const Tensor& b) {
    std::size_t n = a.shape()[0];
    std::size_t k = a.shape()[1];
    std::size_t m = b.shape()[1];
    const Tensor ac = a.detach().contiguous();
    const Tensor bc = b.detach().contiguous();
    Tensor out = Tensor::zeros({n, m});
    const float* A = ac.data();
    const float* B = bc.data();
    float* C = out.data();

    // Строки результата независимы, поэтому их можно считать параллельно.
    // MSVC (OpenMP 2.0) требует знаковый счётчик цикла.
    const std::ptrdiff_t rows = static_cast<std::ptrdiff_t>(n);
    const bool big = n * k * m >= 65536;
    (void)big;
#if defined(_OPENMP)
#pragma omp parallel for if (big)
#endif
    for (std::ptrdiff_t ii = 0; ii < rows; ++ii) {
        std::size_t i = static_cast<std::size_t>(ii);
        for (std::size_t p = 0; p < k; ++p) {
            float a_ip = A[i * k + p];
            for (std::size_t j = 0; j < m; ++j) {
                C[i * m + j] += a_ip * B[p * m + j];
            }
        }
    }
    return out;
}

} // namespace

Tensor matmul(const Tensor& a, const Tensor& b) {
    if (a.ndim() != 2 || b.ndim() != 2) {
        throw std::invalid_argument("matmul(): only 2D tensors are supported, got " +
                                    shape_to_string(a.shape()) + " and " +
                                    shape_to_string(b.shape()));
    }
    if (b.shape()[0] != a.shape()[1]) {
        throw std::invalid_argument("matmul(): shapes " + shape_to_string(a.shape()) + " and " +
                                    shape_to_string(b.shape()) + " are incompatible");
    }
    Tensor out = matmul_values(a, b);
    // C = A·B  =>  dA = dC · Bᵀ,  dB = Aᵀ · dC
    return record_op(out, {a, b}, "MatmulBackward", [a, b](const Tensor& g) {
        Tensor da = a.requires_grad() ? matmul_values(g, b.transpose(0, 1)) : Tensor();
        Tensor db = b.requires_grad() ? matmul_values(a.transpose(0, 1), g) : Tensor();
        return std::vector<Tensor>{da, db};
    });
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
