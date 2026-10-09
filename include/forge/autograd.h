#pragma once
#include <forge/tensor.h>

#include <functional>
#include <string>
#include <vector>

namespace forge {

// Функция обратного прохода: по градиенту выхода возвращает градиенты
// для каждого входа (в том же порядке, что Node::inputs).
// Пустой Tensor() в ответе означает "этому входу градиент не нужен".
using BackwardFn = std::function<std::vector<Tensor>(const Tensor& grad_output)>;

// Узел графа вычислений: "этот тензор получен операцией name из inputs".
struct Node {
    std::string name;            // для отладки: "AddBackward", "MatmulBackward"...
    std::vector<Tensor> inputs;  // входы операции
    BackwardFn backward;         // как посчитать их градиенты
    // Версии данных входов в момент forward: если вход потом изменили,
    // backward посчитал бы градиент не для тех чисел — это ошибка.
    std::vector<std::size_t> saved_versions;

    Node() = default;
    // Разбирает длинную цепочку узлов циклом, а не рекурсией:
    // иначе граф из сотен тысяч операций переполнил бы стек при удалении.
    ~Node();
    Node(const Node&) = delete;
    Node& operator=(const Node&) = delete;
};

// Глобальный (на поток) переключатель: строить ли граф.
class GradMode {
public:
    static bool is_enabled();
    static void set_enabled(bool enabled);
};

// RAII: пока объект жив, граф не строится (как torch.no_grad()).
class NoGradGuard {
public:
    NoGradGuard();
    ~NoGradGuard();
    NoGradGuard(const NoGradGuard&) = delete;
    NoGradGuard& operator=(const NoGradGuard&) = delete;

private:
    bool previous_;
};

// Прикрепляет к результату операции узел графа, если это нужно:
// граф включён и хотя бы одному входу нужен градиент.
Tensor record_op(Tensor out, std::vector<Tensor> inputs, std::string name, BackwardFn backward);

// Обратная операция к broadcasting: суммирует градиент по растянутым осям,
// чтобы его форма совпала с формой исходного тензора.
Tensor sum_to_shape(const Tensor& grad, const Shape& shape);

} // namespace forge
