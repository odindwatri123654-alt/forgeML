#pragma once
#include <forge/tensor.h>

namespace forge {

// --- broadcasting ---
Shape broadcast_shapes(const Shape& a, const Shape& b);

// --- поэлементные операции с двумя тензорами (с broadcasting) ---
Tensor add(const Tensor& a, const Tensor& b);
Tensor sub(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);
Tensor div(const Tensor& a, const Tensor& b);

// --- поэлементные операции с одним тензором ---
Tensor neg(const Tensor& a);
Tensor exp(const Tensor& a);
Tensor log(const Tensor& a);
Tensor relu(const Tensor& a);
Tensor sigmoid(const Tensor& a);
Tensor tanh(const Tensor& a);
Tensor pow(const Tensor& a, float exponent);

// --- редукции ---
Tensor sum(const Tensor& a);                                          // всё -> скаляр
Tensor sum(const Tensor& a, std::size_t dim, bool keepdim = false);
Tensor mean(const Tensor& a);
Tensor mean(const Tensor& a, std::size_t dim, bool keepdim = false);
Tensor max(const Tensor& a, std::size_t dim, bool keepdim = false);
Tensor argmax(const Tensor& a, std::size_t dim);                      // индексы (без градиента)

// --- softmax ---
Tensor softmax(const Tensor& a, std::size_t dim);
Tensor log_softmax(const Tensor& a, std::size_t dim);

// --- линейная алгебра ---
Tensor matmul(const Tensor& a, const Tensor& b);

// --- операторы ---
Tensor operator+(const Tensor& a, const Tensor& b);
Tensor operator-(const Tensor& a, const Tensor& b);
Tensor operator*(const Tensor& a, const Tensor& b);
Tensor operator/(const Tensor& a, const Tensor& b);
Tensor operator-(const Tensor& a);

Tensor operator+(const Tensor& a, float b);
Tensor operator-(const Tensor& a, float b);
Tensor operator*(const Tensor& a, float b);
Tensor operator/(const Tensor& a, float b);
Tensor operator+(float a, const Tensor& b);
Tensor operator-(float a, const Tensor& b);
Tensor operator*(float a, const Tensor& b);
Tensor operator/(float a, const Tensor& b);

} // namespace forge
