#pragma once
#include <forge/tensor.h>

#include <initializer_list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace forge::nn {

// Базовый класс всех слоёв и моделей (как torch.nn.Module).
class Module {
public:
    Module() = default;
    virtual ~Module() = default;
    Module(const Module&) = delete;             // модуль владеет параметрами —
    Module& operator=(const Module&) = delete;  // копировать его случайно опасно

    virtual Tensor forward(const Tensor& x) = 0;
    Tensor operator()(const Tensor& x) { return forward(x); }

    // Все обучаемые параметры: свои + всех вложенных модулей.
    std::vector<Tensor> parameters() const;
    std::vector<std::pair<std::string, Tensor>> named_parameters() const;
    std::size_t num_parameters() const;  // сколько всего чисел обучается

    void zero_grad();

    // Режим обучения / оценки (влияет, например, на Dropout).
    void train(bool on = true);
    void eval() { train(false); }
    bool is_training() const { return training_; }

    // Сохранение и загрузка весов в бинарный файл.
    void save(const std::string& path) const;
    void load(const std::string& path);

protected:
    Tensor register_parameter(const std::string& name, Tensor param);
    void register_module(const std::string& name, std::shared_ptr<Module> module);

private:
    std::vector<std::pair<std::string, Tensor>> params_;
    std::vector<std::pair<std::string, std::shared_ptr<Module>>> children_;
    bool training_ = true;
};

// y = x · W + b,   x: [N, in],  W: [in, out],  b: [out]
class Linear : public Module {
public:
    Linear(std::size_t in_features, std::size_t out_features, bool with_bias = true);
    Tensor forward(const Tensor& x) override;

    Tensor weight;
    Tensor bias;  // пустой Tensor(), если with_bias == false
};

class ReLU : public Module {
public:
    Tensor forward(const Tensor& x) override;
};

class Sigmoid : public Module {
public:
    Tensor forward(const Tensor& x) override;
};

class Tanh : public Module {
public:
    Tensor forward(const Tensor& x) override;
};

// При обучении зануляет каждый элемент с вероятностью p, остальные делит на (1-p).
// В режиме eval() ничего не делает.
class Dropout : public Module {
public:
    explicit Dropout(float p = 0.5f);
    Tensor forward(const Tensor& x) override;

private:
    float p_;
};

// Цепочка модулей: выход одного — вход следующего.
class Sequential : public Module {
public:
    Sequential() = default;
    Sequential(std::initializer_list<std::shared_ptr<Module>> modules);

    void append(std::shared_ptr<Module> module);

    // model.add<nn::Linear>(784, 128) — создать модуль и сразу добавить
    template <typename M, typename... Args>
    std::shared_ptr<M> add(Args&&... args) {
        auto module = std::make_shared<M>(std::forward<Args>(args)...);
        append(module);
        return module;
    }

    Tensor forward(const Tensor& x) override;
    std::size_t size() const { return modules_.size(); }

private:
    std::vector<std::shared_ptr<Module>> modules_;
};

// --- функции потерь ---

// Среднеквадратичная ошибка: mean((pred - target)^2). Формы должны совпадать.
Tensor mse_loss(const Tensor& pred, const Tensor& target);

// Кросс-энтропия для классификации.
// logits: [N, C] — "сырые" оценки классов; targets: [N] — номера классов 0..C-1.
Tensor cross_entropy(const Tensor& logits, const Tensor& targets);

// Доля правильных ответов: argmax(logits) == targets.
float accuracy(const Tensor& logits, const Tensor& targets);

} // namespace forge::nn
