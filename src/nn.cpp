#include <forge/autograd.h>
#include <forge/nn.h>
#include <forge/ops.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>

namespace forge::nn {

// ============================================================
// Module
// ============================================================

Tensor Module::register_parameter(const std::string& name, Tensor param) {
    if (!param.is_leaf()) {
        param = param.detach();
    }
    param.requires_grad_(true);
    params_.emplace_back(name, param);
    return param;
}

void Module::register_module(const std::string& name, std::shared_ptr<Module> module) {
    children_.emplace_back(name, std::move(module));
}

std::vector<std::pair<std::string, Tensor>> Module::named_parameters() const {
    std::vector<std::pair<std::string, Tensor>> result = params_;
    for (const auto& [child_name, child] : children_) {
        for (auto& [name, param] : child->named_parameters()) {
            result.emplace_back(child_name + "." + name, param);
        }
    }
    return result;
}

std::vector<Tensor> Module::parameters() const {
    std::vector<Tensor> result;
    for (auto& [name, param] : named_parameters()) {
        result.push_back(param);
    }
    return result;
}

std::size_t Module::num_parameters() const {
    std::size_t n = 0;
    for (const Tensor& p : parameters()) {
        n += p.numel();
    }
    return n;
}

void Module::zero_grad() {
    for (Tensor& p : parameters()) {
        p.zero_grad();
    }
}

void Module::train(bool on) {
    training_ = on;
    for (auto& [name, child] : children_) {
        child->train(on);
    }
}

// Формат файла:
//   "FGML" | число параметров | для каждого: длина имени, имя, ndim, размеры, числа
namespace {

void write_u64(std::ofstream& out, std::uint64_t v) {
    out.write(reinterpret_cast<const char*>(&v), sizeof(v));
}

std::uint64_t read_u64(std::ifstream& in) {
    std::uint64_t v = 0;
    in.read(reinterpret_cast<char*>(&v), sizeof(v));
    if (!in) {
        throw std::runtime_error("load(): unexpected end of file");
    }
    return v;
}

} // namespace

void Module::save(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("save(): cannot open " + path);
    }
    out.write("FGML", 4);
    auto params = named_parameters();
    write_u64(out, params.size());
    for (const auto& [name, param] : params) {
        write_u64(out, name.size());
        out.write(name.data(), static_cast<std::streamsize>(name.size()));
        write_u64(out, param.ndim());
        for (std::size_t d : param.shape()) {
            write_u64(out, d);
        }
        const Tensor values = param.detach().contiguous();
        out.write(reinterpret_cast<const char*>(values.data()),
                  static_cast<std::streamsize>(values.numel() * sizeof(float)));
    }
}

void Module::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("load(): cannot open " + path);
    }
    char magic[4] = {};
    in.read(magic, 4);
    if (!in || std::string(magic, 4) != "FGML") {
        throw std::runtime_error("load(): " + path + " is not a ForgeML weights file");
    }
    auto params = named_parameters();
    if (read_u64(in) != params.size()) {
        throw std::runtime_error("load(): number of parameters does not match the model");
    }
    // Сначала читаем и проверяем ВЕСЬ файл во временные буферы,
    // и только потом копируем в модель: при любой ошибке модель не меняется.
    std::vector<std::vector<float>> loaded;
    for (auto& [name, param] : params) {
        std::uint64_t name_size = read_u64(in);
        if (name_size > 4096) {
            throw std::runtime_error("load(): corrupted file (parameter name too long)");
        }
        std::string file_name(static_cast<std::size_t>(name_size), '\0');
        in.read(file_name.data(), static_cast<std::streamsize>(file_name.size()));
        std::uint64_t ndim = read_u64(in);
        if (ndim > 64) {
            throw std::runtime_error("load(): corrupted file (too many dimensions)");
        }
        Shape shape(static_cast<std::size_t>(ndim));
        for (std::size_t& d : shape) {
            d = static_cast<std::size_t>(read_u64(in));
        }
        if (file_name != name || shape != param.shape()) {
            throw std::runtime_error("load(): expected " + name + " " +
                                     shape_to_string(param.shape()) + ", found " + file_name +
                                     " " + shape_to_string(shape));
        }
        std::vector<float> values(param.numel());
        in.read(reinterpret_cast<char*>(values.data()),
                static_cast<std::streamsize>(values.size() * sizeof(float)));
        if (!in) {
            throw std::runtime_error("load(): unexpected end of file");
        }
        loaded.push_back(std::move(values));
    }
    // Всё прочитано без ошибок — переносим в параметры (contiguous листья).
    for (std::size_t i = 0; i < params.size(); ++i) {
        std::copy(loaded[i].begin(), loaded[i].end(), params[i].second.data());
    }
}

// ============================================================
// Слои
// ============================================================

Linear::Linear(std::size_t in_features, std::size_t out_features, bool with_bias) {
    if (in_features == 0 || out_features == 0) {
        throw std::invalid_argument("Linear: in_features and out_features must be > 0, got " +
                                    std::to_string(in_features) + " and " +
                                    std::to_string(out_features));
    }
    // Как в PyTorch: веса из U(-k, k), k = 1/sqrt(in_features).
    // Так начальные выходы слоя не слишком большие и не слишком маленькие.
    float k = 1.0f / std::sqrt(static_cast<float>(in_features));
    weight = register_parameter("weight",
                                (Tensor::rand({in_features, out_features}) * 2.0f - 1.0f) * k);
    if (with_bias) {
        bias = register_parameter("bias", (Tensor::rand({out_features}) * 2.0f - 1.0f) * k);
    }
}

Tensor Linear::forward(const Tensor& x) {
    Tensor y = matmul(x, weight);
    return bias.defined() ? y + bias : y;  // bias [out] бродкастится на [N, out]
}

Tensor ReLU::forward(const Tensor& x) { return relu(x); }
Tensor Sigmoid::forward(const Tensor& x) { return sigmoid(x); }
Tensor Tanh::forward(const Tensor& x) { return tanh(x); }

Dropout::Dropout(float p) : p_(p) {
    if (p < 0.0f || p >= 1.0f) {
        throw std::invalid_argument("Dropout: p must be in [0, 1)");
    }
}

Tensor Dropout::forward(const Tensor& x) {
    if (!is_training() || p_ == 0.0f) {
        return x;
    }
    // маска — константа: 0 или 1/(1-p); градиент пройдёт через умножение
    Tensor mask = Tensor::rand(x.shape());
    float* m = mask.data();
    for (std::size_t i = 0; i < mask.numel(); ++i) {
        m[i] = m[i] >= p_ ? 1.0f / (1.0f - p_) : 0.0f;
    }
    return x * mask;
}

Sequential::Sequential(std::initializer_list<std::shared_ptr<Module>> modules) {
    for (const auto& module : modules) {
        append(module);
    }
}

void Sequential::append(std::shared_ptr<Module> module) {
    register_module(std::to_string(modules_.size()), module);
    modules_.push_back(std::move(module));
}

Tensor Sequential::forward(const Tensor& x) {
    Tensor out = x;
    for (auto& module : modules_) {
        out = module->forward(out);
    }
    return out;
}

// ============================================================
// Функции потерь и метрики
// ============================================================

Tensor mse_loss(const Tensor& pred, const Tensor& target) {
    if (pred.shape() != target.shape()) {
        throw std::invalid_argument("mse_loss(): shapes " + shape_to_string(pred.shape()) +
                                    " and " + shape_to_string(target.shape()) + " differ");
    }
    return mean(pow(pred - target, 2.0f));
}

namespace {

void check_classification(const Tensor& logits, const Tensor& targets, const char* fn) {
    if (logits.ndim() != 2 || targets.ndim() != 1 || targets.shape()[0] != logits.shape()[0]) {
        throw std::invalid_argument(std::string(fn) + ": expected logits [N, C] and targets [N], got " +
                                    shape_to_string(logits.shape()) + " and " +
                                    shape_to_string(targets.shape()));
    }
}

} // namespace

Tensor cross_entropy(const Tensor& logits, const Tensor& targets) {
    check_classification(logits, targets, "cross_entropy()");
    std::size_t n = logits.shape()[0];
    std::size_t c = logits.shape()[1];

    // one-hot: в строке i единица стоит в столбце правильного класса
    Tensor one_hot = Tensor::zeros({n, c});
    const Tensor t = targets.detach().contiguous();
    for (std::size_t i = 0; i < n; ++i) {
        float cls = t.data()[i];
        // !(cls >= 0) ловит и отрицательные числа, и NaN
        if (!(cls >= 0.0f) || static_cast<std::size_t>(cls) >= c) {
            throw std::invalid_argument("cross_entropy(): class index " + std::to_string(cls) +
                                        " out of range [0, " + std::to_string(c) + ")");
        }
        if (cls != std::floor(cls)) {
            throw std::invalid_argument("cross_entropy(): class index must be an integer, got " +
                                        std::to_string(cls));
        }
        one_hot.data()[i * c + static_cast<std::size_t>(cls)] = 1.0f;
    }
    // loss = -(1/N) * Σ log p(правильный класс)
    Tensor log_probs = log_softmax(logits, 1);
    return -sum(log_probs * one_hot) / static_cast<float>(n);
}

float accuracy(const Tensor& logits, const Tensor& targets) {
    check_classification(logits, targets, "accuracy()");
    const Tensor predicted = argmax(logits, 1);
    const Tensor t = targets.detach().contiguous();
    std::size_t correct = 0;
    for (std::size_t i = 0; i < predicted.numel(); ++i) {
        if (predicted.data()[i] == t.data()[i]) {
            ++correct;
        }
    }
    return static_cast<float>(correct) / static_cast<float>(predicted.numel());
}

} // namespace forge::nn
