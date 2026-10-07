// День 3: autograd на маленьком примере.
#include <forge/forge.h>

#include <iostream>

using namespace forge;

int main() {
    enable_utf8_console();
    // y = a * b + b^2
    auto a = Tensor::full({}, 2.0f).requires_grad_();
    auto b = Tensor::full({}, 3.0f).requires_grad_();
    auto y = a * b + pow(b, 2.0f);
    std::cout << "y = " << y << "\n";
    std::cout << "y создан операцией: " << y.impl()->grad_fn->name << "\n";

    y.backward();
    std::cout << "dy/da = b        = " << a.grad() << "   (ожидаем 3)\n";
    std::cout << "dy/db = a + 2b   = " << b.grad() << "   (ожидаем 8)\n";

    // Градиент по матрице: L = sum(relu(x · W))
    manual_seed(0);
    auto x = Tensor::randn({2, 3});
    auto W = Tensor::randn({3, 2}).requires_grad_();
    auto L = sum(relu(matmul(x, W)));
    L.backward();
    std::cout << "\nL = " << L << "\n";
    std::cout << "dL/dW =\n" << W.grad() << "\n";

    // В режиме no_grad граф не строится
    {
        NoGradGuard no_grad;
        auto z = W * 2.0f;
        std::cout << "\nпод NoGradGuard requires_grad = " << std::boolalpha << z.requires_grad()
                  << "\n";
    }
}
