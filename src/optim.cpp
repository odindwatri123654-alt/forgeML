#include <forge/optim.h>

#include <cmath>
#include <stdexcept>
#include <utility>

namespace forge::optim {

// ============================================================
// Optimizer
// ============================================================

Optimizer::Optimizer(std::vector<Tensor> params, float lr) : params_(std::move(params)), lr_(lr) {
    for (const Tensor& p : params_) {
        if (!p.requires_grad() || !p.is_leaf() || !p.is_contiguous()) {
            throw std::invalid_argument(
                "Optimizer: parameters must be contiguous leaf tensors with requires_grad");
        }
    }
}

void Optimizer::zero_grad() {
    for (Tensor& p : params_) {
        p.zero_grad();
    }
}

// ============================================================
// SGD
// ============================================================

SGD::SGD(std::vector<Tensor> params, float lr, float momentum, float weight_decay)
    : Optimizer(std::move(params), lr), momentum_(momentum), weight_decay_(weight_decay) {
    for (const Tensor& p : params_) {
        velocity_.emplace_back(p.numel(), 0.0f);
    }
}

void SGD::step() {
    for (std::size_t i = 0; i < params_.size(); ++i) {
        Tensor& p = params_[i];
        Tensor grad = p.grad();
        if (!grad.defined()) {
            continue;  // параметр не участвовал в вычислении loss
        }
        // Пишем прямо в память параметра: это не операция графа.
        float* w = p.data();
        const float* g = grad.data();
        std::vector<float>& v = velocity_[i];
        for (std::size_t j = 0; j < p.numel(); ++j) {
            float d = g[j] + weight_decay_ * w[j];
            if (momentum_ != 0.0f) {
                v[j] = momentum_ * v[j] + d;
                d = v[j];
            }
            w[j] -= lr_ * d;
        }
    }
}

// ============================================================
// Adam
// ============================================================

Adam::Adam(std::vector<Tensor> params, float lr, float beta1, float beta2, float eps,
           float weight_decay)
    : Optimizer(std::move(params), lr),
      beta1_(beta1),
      beta2_(beta2),
      eps_(eps),
      weight_decay_(weight_decay) {
    for (const Tensor& p : params_) {
        m_.emplace_back(p.numel(), 0.0f);
        v_.emplace_back(p.numel(), 0.0f);
    }
}

void Adam::step() {
    ++t_;
    // m и v стартуют с нуля и первые шаги "занижены" — делим на (1 - beta^t)
    const float bias1 = 1.0f - std::pow(beta1_, static_cast<float>(t_));
    const float bias2 = 1.0f - std::pow(beta2_, static_cast<float>(t_));

    for (std::size_t i = 0; i < params_.size(); ++i) {
        Tensor& p = params_[i];
        Tensor grad = p.grad();
        if (!grad.defined()) {
            continue;
        }
        float* w = p.data();
        const float* g = grad.data();
        std::vector<float>& m = m_[i];
        std::vector<float>& v = v_[i];
        for (std::size_t j = 0; j < p.numel(); ++j) {
            float d = g[j] + weight_decay_ * w[j];
            m[j] = beta1_ * m[j] + (1.0f - beta1_) * d;
            v[j] = beta2_ * v[j] + (1.0f - beta2_) * d * d;
            float m_hat = m[j] / bias1;
            float v_hat = v[j] / bias2;
            w[j] -= lr_ * m_hat / (std::sqrt(v_hat) + eps_);
        }
    }
}

} // namespace forge::optim
