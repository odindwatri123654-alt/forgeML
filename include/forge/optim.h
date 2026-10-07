#pragma once
#include <forge/tensor.h>

#include <vector>

namespace forge::optim {

// Базовый класс оптимизаторов: знает список параметров и шаг обучения.
class Optimizer {
public:
    Optimizer(std::vector<Tensor> params, float lr);
    virtual ~Optimizer() = default;

    virtual void step() = 0;  // обновить параметры по их .grad()
    void zero_grad();         // обнулить градиенты всех параметров

    float lr() const { return lr_; }
    void set_lr(float lr) { lr_ = lr; }

protected:
    std::vector<Tensor> params_;
    float lr_;
};

// Стохастический градиентный спуск:
//   g = grad + weight_decay * p
//   v = momentum * v + g          (если momentum > 0)
//   p = p - lr * v
class SGD : public Optimizer {
public:
    SGD(std::vector<Tensor> params, float lr, float momentum = 0.0f, float weight_decay = 0.0f);
    void step() override;

private:
    float momentum_;
    float weight_decay_;
    std::vector<std::vector<float>> velocity_;  // "скорость" для каждого параметра
};

// Adam: адаптивный шаг для каждого числа отдельно.
//   m = b1*m + (1-b1)*g          — среднее градиента
//   v = b2*v + (1-b2)*g^2        — среднее квадрата градиента
//   p = p - lr * m̂ / (sqrt(v̂) + eps),  m̂, v̂ — с поправкой на старт с нуля
class Adam : public Optimizer {
public:
    Adam(std::vector<Tensor> params, float lr = 1e-3f, float beta1 = 0.9f, float beta2 = 0.999f,
         float eps = 1e-8f, float weight_decay = 0.0f);
    void step() override;

private:
    float beta1_;
    float beta2_;
    float eps_;
    float weight_decay_;
    long long t_ = 0;  // номер шага
    std::vector<std::vector<float>> m_;
    std::vector<std::vector<float>> v_;
};

} // namespace forge::optim
