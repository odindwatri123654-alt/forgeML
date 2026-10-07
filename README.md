# ForgeML

**ForgeML** — учебная библиотека глубокого обучения на C++17, повторяющая ядро PyTorch:
тензоры со strides и views, автоматическое дифференцирование (autograd), слои нейросетей,
оптимизаторы и загрузку данных. Без внешних зависимостей.

```cpp
#include <forge/forge.h>
using namespace forge;

nn::Sequential model;
model.add<nn::Linear>(784, 128);
model.add<nn::ReLU>();
model.add<nn::Linear>(128, 10);

optim::Adam optimizer(model.parameters(), 1e-3f);

optimizer.zero_grad();
Tensor loss = nn::cross_entropy(model(images), labels);
loss.backward();
optimizer.step();
```

Двухслойная сеть из примера `mnist` за 5 эпох достигает **≈97.6 %** точности на тестовой
выборке MNIST (≈4 с на эпоху в Release-сборке на 4 ядрах).

---

## Содержание

1. [Возможности](#возможности)
2. [Структура проекта](#структура-проекта)
3. [Сборка](#сборка)
4. [Запуск примеров](#запуск-примеров)
5. [Тесты](#тесты)
6. [Руководство по API](#руководство-по-api)
7. [Как устроено внутри](#как-устроено-внутри)
8. [Как расширять библиотеку](#как-расширять-библиотеку)
9. [Отличия от PyTorch](#отличия-от-pytorch)
10. [Частые проблемы](#частые-проблемы)

---

## Возможности

| Модуль | Что есть |
|---|---|
| `tensor.h` | `Tensor`: фабрики (`zeros`, `ones`, `full`, `arange`, `randn`, `rand`, `eye`, `from_vector`), доступ `at`/`item`/`data`, views без копирования (`view`, `reshape`, `transpose`, `permute`, `expand`, `unsqueeze`, `squeeze`), `contiguous`, `clone`, печать |
| `ops.h` | `+ - * /` с broadcasting, `neg`, `exp`, `log`, `pow`, `relu`, `sigmoid`, `tanh`, `sum`/`mean`/`max` (всё или по оси, `keepdim`), `argmax`, `softmax`, `log_softmax`, `matmul` (OpenMP) |
| `autograd.h` | граф вычислений, `backward()`, накопление градиентов, `NoGradGuard`, `detach()` |
| `nn.h` | `Module`, `Linear`, `ReLU`, `Sigmoid`, `Tanh`, `Dropout`, `Sequential`, `mse_loss`, `cross_entropy`, `accuracy`, сохранение/загрузка весов |
| `optim.h` | `SGD` (momentum, weight decay), `Adam` |
| `data.h` | `Dataset`, `DataLoader` (батчи + перемешивание), `load_mnist` |
| `console.h` | `enable_utf8_console()` — русский текст в консоли Windows |

## Структура проекта

```
forgeml/
├── CMakeLists.txt            сборка
├── include/forge/            публичные заголовки
│   ├── forge.h               подключает всё сразу
│   ├── tensor.h              Tensor, TensorImpl, Storage
│   ├── autograd.h            Node, GradMode, NoGradGuard, record_op, sum_to_shape
│   ├── ops.h                 математические операции
│   ├── nn.h                  слои, функции потерь
│   ├── optim.h               оптимизаторы
│   ├── data.h                Dataset, DataLoader, MNIST
│   └── console.h             UTF-8 в консоли Windows
├── src/                      реализация (.cpp для каждого заголовка)
├── examples/
│   ├── playground.cpp        тензоры, views, broadcasting, matmul
│   ├── autograd_demo.cpp     градиенты на маленьком примере
│   ├── xor.cpp               сеть выучивает XOR
│   ├── regression.cpp        сеть приближает sin(x)
│   └── mnist.cpp             распознавание цифр MNIST
├── tests/                    41 тест, включая численную проверку всех градиентов
├── scripts/                  скачивание MNIST (PowerShell и bash)
└── docs/ForgeML_reference.docx   справочник по каждому классу и методу
```

## Сборка

Нужны: компилятор C++17 (MSVC 2019+, GCC 9+, Clang 10+) и CMake 3.20+.
Внешних библиотек нет; OpenMP используется, если компилятор его поддерживает.

### Windows + VS Code (рекомендуется)

1. Установите **Build Tools for Visual Studio 2022** с нагрузкой «Desktop development with C++».
2. В VS Code установите расширения **C/C++** и **CMake Tools**.
3. Откройте **корень** проекта: `File → Open Folder → forgeml`.
4. `Ctrl+Shift+P` → **CMake: Select a Kit** → `Visual Studio Build Tools 2022 Release - amd64`.
5. `Ctrl+Shift+P` → **CMake: Select Variant** → **Release** (для MNIST важно: Debug в 10–50 раз медленнее).
6. `Ctrl+Shift+P` → **CMake: Configure**, затем **F7** — сборка.
7. `Ctrl+Shift+P` → **CMake: Set Launch/Debug Target** → нужный пример → **Shift+F5** — запуск.

> Кнопка ▷ **Run Code** (расширение Code Runner) собирает только один файл и для проекта
> не подходит. Пользуйтесь кнопками CMake Tools в строке состояния.

### Windows из командной строки

В **Developer PowerShell for VS 2022**, из корня проекта:

```powershell
cmake -S . -B build
cmake --build build --config Release
.\build\Release\forge_tests.exe
```

### Linux / macOS

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/forge_tests
```

### Опции CMake

| Опция | По умолчанию | Смысл |
|---|---|---|
| `FORGE_BUILD_TESTS` | `ON` | собирать `forge_tests` |
| `FORGE_BUILD_EXAMPLES` | `ON` | собирать примеры |
| `FORGE_USE_OPENMP` | `ON` | параллелить `matmul` по ядрам процессора |

Пример: `cmake -S . -B build -DFORGE_USE_OPENMP=OFF`.

### Использование в своём проекте

```cmake
add_subdirectory(forgeml)
target_link_libraries(my_app PRIVATE forge)
```

## Запуск примеров

Пути ниже — для Windows (`build\Release\...`). На Linux/macOS — `./build/...`.

| Пример | Команда | Что увидите |
|---|---|---|
| playground | `build\Release\playground.exe` | views, broadcasting, редукции, matmul |
| autograd_demo | `build\Release\autograd_demo.exe` | `dy/da = 3`, `dy/db = 8`, градиент матрицы |
| xor | `build\Release\xor.exe` | loss падает до ~3e-5, предсказания ≈ 0 1 1 0 |
| regression | `build\Release\regression.exe` | MSE ~1e-6, таблица sin(x) против модели |
| mnist | `build\Release\mnist.exe [папка] [эпохи]` | точность ≈97.6 % за 5 эпох |

### MNIST

Сначала скачайте данные (≈11 МБ) в папку `data/`:

```powershell
# Windows, из корня проекта
powershell -ExecutionPolicy Bypass -File scripts\download_mnist.ps1
```
```bash
# Linux / macOS / Git Bash
bash scripts/download_mnist.sh
```

Затем:

```powershell
.\build\Release\mnist.exe            # данные из <корень>/data, 5 эпох
.\build\Release\mnist.exe data 10    # своя папка и 10 эпох
```

Ожидаемый вывод:

```
train: 60000 картинок, test: 10000
параметров в модели: 101770

эпоха 1  loss 0.347  точность на test 94.4%
эпоха 2  loss 0.159  точность на test 96.1%
эпоха 3  loss 0.111  точность на test 96.6%
эпоха 4  loss 0.084  точность на test 97.2%
эпоха 5  loss 0.066  точность на test 97.6%

веса сохранены в mnist_mlp.bin
```

## Тесты

```powershell
.\build\Release\forge_tests.exe     # или: ctest --test-dir build -C Release
```

Тесты используют собственный мини-фреймворк (`tests/test_framework.h`, ~80 строк) — ничего
скачивать не нужно. Группы тестов:

- `tensor_*` — память, strides, views, печать, ошибки;
- `ops_*` — операции, broadcasting, редукции, softmax, matmul;
- `autograd_*` и `gradcheck_*` — **каждый** градиент сравнивается с численной производной
  `(f(x+ε) − f(x−ε)) / 2ε`;
- `nn_*`, `optim_*`, `train_xor`, `save_and_load_roundtrip`, `dataloader_batches`.

Добавить тест — написать в любом файле из `tests/`:

```cpp
TEST(my_feature) {
    CHECK(1 + 1 == 2);
    CHECK_NEAR(std::sqrt(2.0), 1.4142, 1e-4);
    CHECK_THROWS(Tensor::from_vector({1, 2, 3}, {2, 2}));
}
```

## Руководство по API

Всё подключается одной строкой `#include <forge/forge.h>`. Пространства имён:
`forge` (тензоры, операции, autograd), `forge::nn`, `forge::optim`, `forge::data`.

### Тензоры

```cpp
auto a = Tensor::zeros({2, 3});
auto b = Tensor::from_vector({1, 2, 3, 4, 5, 6}, {2, 3});
auto r = Tensor::randn({3, 3});              // N(0, 1)
auto u = Tensor::rand({3});                  // U[0, 1)
manual_seed(42);                             // воспроизводимые случайные числа

b.at({1, 0});              // 4 — чтение
b.at({1, 0}) = 10;         // запись
b.shape();                 // {2, 3}
b.numel();                 // 6
Tensor::full({}, 5).item();// 5 — значение скаляра
std::cout << b << "\n";    // tensor([[1, 2, 3], [10, 5, 6]])
```

**Копия `Tensor` — это тот же тензор** (как в PyTorch): `Tensor c = b;` не копирует данные.
Независимая копия — `b.clone()`.

### Views (без копирования данных)

```cpp
auto m = Tensor::arange(0, 6).view({2, 3});   // та же память, другая форма
auto t = m.transpose(0, 1);                    // [3, 2], данные не тронуты
t.is_contiguous();                             // false
t.reshape({6});                                // view, если можно, иначе копия
m.permute({1, 0});                             // произвольная перестановка осей
Tensor::ones({3}).expand({4, 3});              // растянуть через stride = 0
Tensor::ones({3}).unsqueeze(0);                // [1, 3]
```

### Операции

```cpp
x + y;  x - 1.0f;  2.0f * x;  x / y;  -x;      // поэлементно, с broadcasting
matmul(A, B);                                  // [n,k] · [k,m] -> [n,m]
relu(x); sigmoid(x); tanh(x); exp(x); log(x); pow(x, 2.0f);
sum(x); sum(x, 1); sum(x, 1, /*keepdim=*/true);
mean(x, 0); max(x, 1); argmax(x, 1);
softmax(x, 1); log_softmax(x, 1);
```

Broadcasting работает по правилам NumPy: формы выравниваются по правому краю, ось размера 1
растягивается. `[64, 128] + [128]` → `[64, 128]`.

### Autograd

```cpp
auto w = Tensor::randn({3, 2}).requires_grad_();  // лист, за которым следим
auto loss = sum(relu(matmul(x, w)));
loss.backward();          // посчитать d(loss)/d(w)
w.grad();                 // градиент той же формы, что w
w.zero_grad();            // градиенты НАКАПЛИВАЮТСЯ — обнуляйте перед каждым шагом

{
    NoGradGuard no_grad;  // внутри блока граф не строится (быстрее, меньше памяти)
    auto prediction = model(x);
}
auto frozen = w.detach(); // те же данные, но вне графа
```

`backward()` без аргумента работает только для скаляра; для тензора передайте градиент
выхода: `y.backward(Tensor::ones(y.shape()))`.

### Слои и модели

```cpp
nn::Sequential model;
model.add<nn::Linear>(784, 128);   // y = x·W + b, W: [784, 128]
model.add<nn::ReLU>();
model.add<nn::Dropout>(0.2f);
model.add<nn::Linear>(128, 10);

Tensor logits = model(x);          // x: [batch, 784] -> [batch, 10]
model.parameters();                // все обучаемые тензоры
model.named_parameters();          // {"0.weight", ...}, {"0.bias", ...}, ...
model.num_parameters();            // 101770 (у ReLU и Dropout параметров нет)
model.eval();  model.train();      // режимы (влияет на Dropout)
model.save("model.bin");  model.load("model.bin");
```

Свой модуль:

```cpp
class MLP : public nn::Module {
public:
    MLP() {
        fc1 = std::make_shared<nn::Linear>(784, 256);
        fc2 = std::make_shared<nn::Linear>(256, 10);
        register_module("fc1", fc1);
        register_module("fc2", fc2);
    }
    Tensor forward(const Tensor& x) override {
        return (*fc2)(relu((*fc1)(x)));
    }
private:
    std::shared_ptr<nn::Linear> fc1, fc2;
};
```

Собственный обучаемый параметр — `weight = register_parameter("weight", Tensor::randn({n}));`.

### Функции потерь

```cpp
nn::mse_loss(pred, target);          // регрессия, формы должны совпадать
nn::cross_entropy(logits, labels);   // классификация: logits [N, C], labels [N] (номера классов)
nn::accuracy(logits, labels);        // доля правильных ответов, float
```

### Оптимизаторы

```cpp
optim::SGD  sgd(model.parameters(), /*lr=*/0.1f, /*momentum=*/0.9f, /*weight_decay=*/0.0f);
optim::Adam adam(model.parameters(), /*lr=*/1e-3f);
adam.set_lr(adam.lr() * 0.5f);       // ручное расписание learning rate
```

### Данные

```cpp
data::Dataset train = data::load_mnist("data/train-images-idx3-ubyte",
                                       "data/train-labels-idx1-ubyte");
data::DataLoader loader(train, /*batch_size=*/64, /*shuffle=*/true);

for (int epoch = 0; epoch < 5; ++epoch) {
    loader.reshuffle();
    for (std::size_t b = 0; b < loader.num_batches(); ++b) {
        auto [x, y] = loader.batch(b);
        // ...
    }
}
```

Свой датасет: `data::Dataset ds{inputs, targets};` — первая ось обоих тензоров — номер примера.

### Полный цикл обучения

```cpp
for (std::size_t b = 0; b < loader.num_batches(); ++b) {
    auto [x, y] = loader.batch(b);
    optimizer.zero_grad();                         // 1. обнулить градиенты
    Tensor loss = nn::cross_entropy(model(x), y);  // 2. прямой проход
    loss.backward();                               // 3. обратный проход
    optimizer.step();                              // 4. шаг оптимизатора
}
```

## Как устроено внутри

### Тензор: три слоя

```
Tensor (handle)  ──shared_ptr──►  TensorImpl  ──shared_ptr──►  Storage
  копируется                       shape, strides, offset,       std::vector<float>
  как указатель                    requires_grad, grad, grad_fn
```

Адрес элемента: `offset + Σ index[d] · strides[d]`. Views меняют только shape/strides/offset,
поэтому `transpose`, `view`, `expand` ничего не копируют. `expand` использует stride = 0 —
на этом построен broadcasting.

### Autograd

Каждая операция вызывает `record_op(...)`: если хотя бы одному входу нужен градиент,
к результату прикрепляется `Node` — имя операции, её входы и лямбда `backward`,
которая по градиенту выхода возвращает градиенты входов.

`loss.backward()`:

1. обходом в глубину строит топологический порядок графа;
2. идёт от `loss` к листьям, вызывая `Node::backward` и **суммируя** градиенты тензоров,
   использованных несколько раз;
3. у листьев (параметров) накапливает результат в `.grad()`.

Градиенты broadcasting сворачиваются функцией `sum_to_shape`. Обратный проход выполняется
под `NoGradGuard`, поэтому сам в граф не попадает.

### Производительность

- Все операции работают с плоскими contiguous-массивами; сложность strides собрана в `clone()`.
- `matmul` использует порядок циклов i-k-j (дружелюбный к кэшу) и OpenMP по строкам.
- Release-сборка обязательна для обучения: в Debug нет оптимизаций и векторизации.

## Как расширять библиотеку

**Новая операция с градиентом** (в `src/ops.cpp`, объявление — в `include/forge/ops.h`):

```cpp
Tensor softplus(const Tensor& a) {
    // 1. значения
    Tensor out = unary_op(a, [](float x) { return std::log1p(std::exp(x)); });
    // 2. производная: softplus'(x) = sigmoid(x)
    return record_op(out, {a}, "SoftplusBackward", [a](const Tensor& g) {
        return std::vector<Tensor>{g * sigmoid(a)};
    });
}
```

И обязательно тест: `gradcheck([](auto& in) { return softplus(in[0]); }, {Tensor::randn({2, 3})});`
в `tests/test_autograd.cpp`.

**Новый слой** — наследник `nn::Module` с методом `forward` (см. пример `MLP` выше).

**Новый оптимизатор** — наследник `optim::Optimizer` с методом `step()` (см. `src/optim.cpp`).

## Отличия от PyTorch

| | PyTorch | ForgeML |
|---|---|---|
| Типы данных | float16/32/64, int, bool… | только `float` |
| Устройства | CPU, CUDA, MPS | только CPU |
| Вес `Linear` | `[out, in]`, `y = x·Wᵀ + b` | `[in, out]`, `y = x·W + b` |
| `matmul` | батчи, векторы | только 2D матрицы |
| Метки классов | `int64` тензор | `float` тензор с номерами классов |
| `max(dim)` | возвращает (values, indices) | только values; индексы — `argmax` |
| Градиент `max` при равных элементах | в один элемент | во все равные элементы |
| Граф после `backward` | освобождается | живёт, пока жив выход |
| Производные высших порядков | есть | нет |

## Частые проблемы

| Симптом | Причина и решение |
|---|---|
| «Кракозябры» вместо русского текста | консоль Windows не в UTF-8. Примеры вызывают `enable_utf8_console()`; в своём `main` сделайте так же |
| `"g++" не является внутренней или внешней командой` | запуск через Code Runner. Собирайте через CMake Tools (F7) |
| `unresolved external symbol forge::...` / `undefined reference` | новый `.cpp` не добавлен в `add_library` в `CMakeLists.txt` |
| `cannot open .../train-images-idx3-ubyte` | не скачан MNIST — запустите `scripts/download_mnist.ps1` |
| MNIST обучается очень медленно | собрана Debug-версия. Выберите вариант **Release** |
| `Tensor is undefined` | обращение к пустому тензору: например, `w.grad()` до `backward()` или после `zero_grad()` |
| `backward(): output has N elements` | `backward()` без аргумента — только для скаляра; сверните loss через `sum`/`mean` |
| `broadcast_shapes(): sizes 3 and 2 are incompatible` | формы не подходят для broadcasting; проверьте `shape()` операндов |
| `view(): tensor is not contiguous` | после `transpose`/`permute` используйте `reshape` |
| Loss не уменьшается | забыт `optimizer.zero_grad()` (градиенты копятся) или слишком большой learning rate |

## Документация

Подробный справочник по каждому классу, методу и приёму C++, разбитый по дням разработки:
[`docs/ForgeML_reference.docx`](docs/ForgeML_reference.docx).
