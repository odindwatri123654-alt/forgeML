#include "qa.h"

#include <forge/forge.h>

#include <chrono>
#include <fstream>
#include <memory>
#include <thread>

using namespace forge;

namespace {

// Резидентная память процесса в МБ (Linux); на других системах — 0.
double rss_mb() {
#ifdef __linux__
    std::ifstream f("/proc/self/statm");
    long pages = 0, resident = 0;
    f >> pages >> resident;
    return double(resident) * 4096.0 / (1024.0 * 1024.0);
#else
    return 0.0;
#endif
}

void deep_chain(std::size_t depth) {
    auto x = Tensor::full({}, 1.0f).requires_grad_();
    Tensor y = x;
    for (std::size_t i = 0; i < depth; ++i) y = y + 1.0f;
    y.backward();
    EXPECT(x.grad().item() == 1.0f, "dy/dx = 1");
}

double seconds_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}

} // namespace

QA_TEST(deep_graph_10k, "12. Нагрузка", "граф из 10 000 операций подряд (как длинная RNN): backward") {
    deep_chain(10000);
}

QA_TEST(deep_graph_100k, "12. Нагрузка", "граф из 100 000 операций подряд: backward") {
    deep_chain(100000);
}

QA_TEST(deep_graph_1m_forward_only, "12. Нагрузка",
        "1 000 000 операций подряд без backward: граф должен просто удалиться") {
    auto x = Tensor::full({}, 1.0f).requires_grad_();
    Tensor y = x;
    for (int i = 0; i < 1000000; ++i) y = y + 1.0f;
    EXPECT(y.item() == 1000001.0f, "значение 1000001");
}

QA_TEST(no_memory_leak_in_training, "12. Нагрузка",
        "3000 шагов обучения: память не растёт (граф каждого шага освобождается)") {
    manual_seed(1);
    nn::Sequential m;
    m.add<nn::Linear>(64, 64);
    m.add<nn::ReLU>();
    m.add<nn::Linear>(64, 10);
    optim::Adam opt(m.parameters(), 1e-3f);
    auto x = Tensor::randn({32, 64});
    auto y = Tensor::zeros({32});
    auto step = [&] {
        opt.zero_grad();
        nn::cross_entropy(m(x), y).backward();
        opt.step();
    };
    for (int i = 0; i < 200; ++i) step();
    double before = rss_mb();
    for (int i = 0; i < 3000; ++i) step();
    double growth = rss_mb() - before;
    EXPECT(growth < 5.0, "рост памяти < 5 МБ (получено " + qa::str(growth) + " МБ)");
}

QA_TEST(graph_freed_after_loss_dies, "12. Нагрузка",
        "промежуточные тензоры удаляются, когда loss выходит из области видимости") {
    std::weak_ptr<TensorImpl> hidden;
    auto w = Tensor::randn({100, 100}).requires_grad_();
    {
        auto h = relu(matmul(w, w));
        hidden = h.impl();
        auto loss = sum(h);
        loss.backward();
    }
    EXPECT(hidden.expired(), "промежуточный тензор освобождён");
}

QA_TEST(matmul_speed_512, "12. Нагрузка", "matmul 512×512·512×512 быстрее 0.5 с (Release)") {
    auto a = Tensor::randn({512, 512});
    auto b = Tensor::randn({512, 512});
    matmul(a, b);  // прогрев
    auto t = std::chrono::steady_clock::now();
    matmul(a, b);
    double s = seconds_since(t);
    EXPECT(s < 0.5, "время " + qa::str(s) + " с");
}

QA_TEST(big_batch_forward, "12. Нагрузка",
        "прямой проход MLP на 10 000 × 784 (весь test MNIST за раз) быстрее 2 с") {
    nn::Sequential m;
    m.add<nn::Linear>(784, 128);
    m.add<nn::ReLU>();
    m.add<nn::Linear>(128, 10);
    auto x = Tensor::rand({10000, 784});
    NoGradGuard ng;
    auto t = std::chrono::steady_clock::now();
    auto out = m(x);
    EXPECT(out.shape() == (Shape{10000, 10}), "форма [10000, 10]");
    EXPECT(seconds_since(t) < 2.0, "время " + qa::str(seconds_since(t)) + " с");
}

QA_TEST(no_grad_is_per_thread, "13. Потоки",
        "NoGradGuard в одном потоке не выключает граф в другом") {
    auto x = Tensor::ones({1}).requires_grad_();
    bool other_thread_tracks = false;
    {
        NoGradGuard ng;
        std::thread t([&] { other_thread_tracks = (x * 2.0f).requires_grad(); });
        t.join();
    }
    EXPECT(other_thread_tracks, "в другом потоке граф включён");
}

QA_TEST(parallel_independent_training, "13. Потоки",
        "4 потока обучают свои модели одновременно: все сходятся, без падений") {
    std::vector<float> losses(4, 1e9f);
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([t, &losses] {
            nn::Sequential m;
            m.add<nn::Linear>(2, 16);
            m.add<nn::Tanh>();
            m.add<nn::Linear>(16, 1);
            optim::Adam opt(m.parameters(), 0.02f);
            auto x = Tensor::randn({64, 2});
            auto y = sum(x * x, 1, true);
            for (int i = 0; i < 300; ++i) {
                opt.zero_grad();
                Tensor loss = nn::mse_loss(m(x), y);
                loss.backward();
                opt.step();
                losses[std::size_t(t)] = loss.item();
            }
        });
    }
    for (auto& th : threads) th.join();
    for (float l : losses) EXPECT(l < 0.1f, "loss каждого потока < 0.1 (" + qa::str(l) + ")");
}
