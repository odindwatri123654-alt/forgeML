#include <forge/ops.h>
#include <forge/tensor.h>
#include <iostream>

using namespace forge;

int main() {
    std::cout << "=== 1. view / reshape ===\n";
    auto a = Tensor::arange(0, 6);
    auto m = a.view({2, 3});
    std::cout << m << "\n";
    m.at({0, 0}) = 100;
    std::cout << "a after m[0][0]=100: " << a << "\n";

    std::cout << "\n=== 2. transpose ===\n";
    auto t = m.transpose(0, 1);
    std::cout << t << "\n";
    std::cout << "shape: " << t.shape()[0] << "x" << t.shape()[1]
              << ", strides: [" << t.strides()[0] << ", " << t.strides()[1] << "]"
              << ", contiguous: " << std::boolalpha << t.is_contiguous() << "\n";
    auto tc = t.contiguous();
    std::cout << "after contiguous(): strides [" << tc.strides()[0] << ", " << tc.strides()[1]
              << "], contiguous: " << tc.is_contiguous() << "\n";

    std::cout << "\n=== 3. broadcasting ===\n";
    auto x = Tensor::from_vector({1, 2, 3, 4, 5, 6}, {2, 3});
    auto bias = Tensor::from_vector({10, 20, 30}, {3});
    std::cout << x + bias << "\n";
    auto col = Tensor::from_vector({100, 200}, {2, 1});
    std::cout << x + col << "\n";
    std::cout << x * 2.0f << "\n";

    std::cout << "\n=== 4. reductions ===\n";
    std::cout << "sum all: " << sum(x) << "\n";
    std::cout << "sum dim0: " << sum(x, 0) << "\n";
    std::cout << "sum dim1: " << sum(x, 1) << "\n";
    std::cout << "sum dim1 keepdim: " << sum(x, 1, true) << "\n";
    std::cout << "mean dim1: " << mean(x, 1) << "\n";
    std::cout << "max dim0: " << max(x, 0) << "\n";

    std::cout << "\n=== 5. matmul ===\n";
    auto A = Tensor::from_vector({1, 2, 3, 4, 5, 6}, {2, 3});
    auto B = Tensor::from_vector({7, 8, 9, 10, 11, 12}, {3, 2});
    std::cout << matmul(A, B) << "\n";
    std::cout << matmul(A, A.transpose(0, 1)) << "\n";
    std::cout << matmul(Tensor::eye(2), Tensor::from_vector({5, 6, 7, 8}, {2, 2})) << "\n";

    std::cout << "\n=== 6. errors ===\n";
    try {
        x + Tensor::ones({2});
    } catch (const std::exception& e) {
        std::cout << "error: " << e.what() << "\n";
    }
    try {
        t.view({6});
    } catch (const std::exception& e) {
        std::cout << "error: " << e.what() << "\n";
    }
    std::cout << "reshape works: " << t.reshape({6}) << "\n";
    std::cout << "relu: " << relu(Tensor::from_vector({-2, -1, 0, 1, 2}, {5})) << "\n";
    std::cout << "1 - x: " << (1.0f - x) << "\n";
}
