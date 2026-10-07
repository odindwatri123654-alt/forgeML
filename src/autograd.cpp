#include <forge/autograd.h>
#include <forge/ops.h>

#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace forge {

// ============================================================
// GradMode и NoGradGuard
// ============================================================

namespace {

bool& grad_enabled_flag() {
    thread_local bool enabled = true;  // у каждого потока свой флаг
    return enabled;
}

} // namespace

bool GradMode::is_enabled() { return grad_enabled_flag(); }
void GradMode::set_enabled(bool enabled) { grad_enabled_flag() = enabled; }

NoGradGuard::NoGradGuard() : previous_(GradMode::is_enabled()) { GradMode::set_enabled(false); }
NoGradGuard::~NoGradGuard() { GradMode::set_enabled(previous_); }

// ============================================================
// Запись операции в граф
// ============================================================

Tensor record_op(Tensor out, std::vector<Tensor> inputs, std::string name, BackwardFn backward) {
    if (!GradMode::is_enabled()) {
        return out;
    }
    bool any_requires_grad = false;
    for (const Tensor& input : inputs) {
        if (input.requires_grad()) {
            any_requires_grad = true;
            break;
        }
    }
    if (!any_requires_grad) {
        return out;
    }
    auto node = std::make_shared<Node>();
    node->name = std::move(name);
    node->inputs = std::move(inputs);
    node->backward = std::move(backward);

    out.impl()->requires_grad = true;
    out.impl()->grad_fn = std::move(node);
    return out;
}

Tensor sum_to_shape(const Tensor& grad, const Shape& shape) {
    Tensor g = grad;
    // 1. лишние оси слева (их добавил broadcasting) — суммируем целиком
    while (g.ndim() > shape.size()) {
        g = sum(g, 0);
    }
    // 2. оси, которые были размером 1, а стали больше — суммируем, оставляя 1
    for (std::size_t d = 0; d < shape.size(); ++d) {
        if (shape[d] == 1 && g.shape()[d] != 1) {
            g = sum(g, d, /*keepdim=*/true);
        }
    }
    return g;
}

// ============================================================
// backward: обратный проход по графу
// ============================================================

namespace {

// Обход в глубину: кладём узел в order только ПОСЛЕ всех его входов.
// Тогда в обратном порядке каждый узел идёт раньше своих входов.
void build_topo(TensorImpl* impl, std::unordered_set<TensorImpl*>& visited,
                std::vector<TensorImpl*>& order) {
    if (!visited.insert(impl).second) {
        return;  // уже были здесь (тензор использован в графе несколько раз)
    }
    if (impl->grad_fn) {
        for (const Tensor& input : impl->grad_fn->inputs) {
            if (input.requires_grad()) {
                build_topo(input.impl().get(), visited, order);
            }
        }
    }
    order.push_back(impl);
}

} // namespace

void Tensor::backward() const {
    if (numel() != 1) {
        throw std::invalid_argument(
            "backward(): output has " + std::to_string(numel()) +
            " elements; call backward(grad) with an explicit gradient");
    }
    backward(Tensor::ones(shape()));
}

void Tensor::backward(const Tensor& grad) const {
    if (!requires_grad()) {
        throw std::runtime_error("backward(): tensor does not require grad");
    }
    if (grad.shape() != shape()) {
        throw std::invalid_argument("backward(): grad shape " + shape_to_string(grad.shape()) +
                                    " does not match tensor shape " + shape_to_string(shape()));
    }
    NoGradGuard no_grad;  // сам обратный проход в граф не записываем

    // 1. Топологический порядок всех тензоров, от которых зависит this.
    std::unordered_set<TensorImpl*> visited;
    std::vector<TensorImpl*> order;
    build_topo(impl_.get(), visited, order);

    // 2. Градиенты промежуточных тензоров живут только во время backward.
    std::unordered_map<TensorImpl*, Tensor> grads;
    grads.emplace(impl_.get(), grad);

    // 3. Идём от выхода к входам.
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        TensorImpl* impl = *it;
        auto found = grads.find(impl);
        if (found == grads.end()) {
            continue;
        }
        Tensor g = found->second;

        if (!impl->grad_fn) {
            // Лист: накапливаем градиент (+=), как в PyTorch.
            if (!impl->grad) {
                impl->grad = g.clone().impl();
            } else {
                impl->grad = (Tensor(impl->grad) + g).impl();
            }
            continue;
        }

        const Node& node = *impl->grad_fn;
        std::vector<Tensor> input_grads = node.backward(g);
        if (input_grads.size() != node.inputs.size()) {
            throw std::logic_error(node.name + ": returned wrong number of gradients");
        }
        for (std::size_t i = 0; i < node.inputs.size(); ++i) {
            const Tensor& input = node.inputs[i];
            const Tensor& input_grad = input_grads[i];
            if (!input.requires_grad() || !input_grad.defined()) {
                continue;
            }
            if (input_grad.shape() != input.shape()) {
                throw std::logic_error(node.name + ": gradient shape " +
                                       shape_to_string(input_grad.shape()) +
                                       " does not match input shape " +
                                       shape_to_string(input.shape()));
            }
            auto [slot, inserted] = grads.emplace(input.impl().get(), input_grad);
            if (!inserted) {
                slot->second = slot->second + input_grad;  // тензор использован несколько раз
            }
        }
    }
}

} // namespace forge
