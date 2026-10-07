module.exports = ({ p, h1, h2, h3, bullets, code, entry, table }) => [
  h1('День 4. Autograd: все производные и их проверка'),
  p('Цель дня: у каждой операции и каждого view есть правильный backward, и это доказано численной проверкой (gradcheck).'),
  h2('Что появилось в этот день'),
  table(['Сущность', 'Вид', 'Файл'], [
    ['`sum_to_shape`', 'свободная функция', 'autograd.cpp'],
    ['backward для `exp`, `log`, `pow`, `relu`, `sigmoid`, `tanh`', 'формулы', 'ops.cpp'],
    ['backward для `matmul`', 'формула', 'ops.cpp'],
    ['backward для `max`; `argmax`', 'формула / функция', 'ops.cpp'],
    ['`softmax`, `log_softmax`', 'функции', 'ops.cpp'],
    ['backward для `view`, `transpose`, `permute`, `expand`, `clone`', 'формулы', 'tensor.cpp'],
    ['`DimSplit`, `split_at`, `reduced_shape`', 'внутренние', 'ops.cpp'],
    ['`gradcheck`', 'тестовая функция', 'tests/test_autograd.cpp'],
  ], [5200, 2200, 2200]),

  h2('Broadcasting в обратную сторону'),
  ...entry({
    name: '`sum_to_shape(grad, shape)`', file: 'src/autograd.cpp',
    sig: ['Tensor sum_to_shape(const Tensor& grad, const Shape& shape);'],
    what: 'Сворачивает градиент до формы исходного тензора. Обратная операция к broadcasting.',
    how: [
      'Почему сумма: при broadcasting один элемент bias использовался в каждой строке батча. Его вклад в loss — сумма вкладов по всем строкам, значит и градиент — сумма.',
      'Шаг 1: пока у градиента больше осей, чем у формы — `sum(g, 0)` (убрать оси, добавленные слева).',
      'Шаг 2: оси, которые в исходной форме были размера 1, а у градиента больше — `sum(g, d, keepdim=true)`.',
    ],
    example: ['// x: [64, 128], bias: [128], y = x + bias', '// g: [64, 128]  ->  sum_to_shape(g, {128}) = sum(g, 0): [128]'],
  }),

  h2('Производные поэлементных функций'),
  table(['Функция', 'Производная', 'Как в коде'], [
    ['eˣ', 'eˣ', '`g * exp(a)`'],
    ['ln x', '1/x', '`g / a`'],
    ['xᵖ', 'p·xᵖ⁻¹', '`g * exponent * pow(a, exponent - 1)`'],
    ['relu(x)', '1 при x > 0, иначе 0', '`g * mask`, mask = (a > 0)'],
    ['sigmoid(x) = s', 's·(1 − s)', '`g * s * (1 - s)`'],
    ['tanh(x) = t', '1 − t²', '`g * (1 - t * t)`'],
  ], [2200, 2600, 4800]),
  p('Во всех случаях значение функции пересчитывается из входа `a` внутри лямбды (например, `sigmoid(a)`), а не берётся из выхода: узел не должен держать ссылку на собственный результат (иначе цикл shared_ptr → утечка памяти).'),

  h2('Производная матричного умножения'),
  ...entry({
    name: '`matmul` — backward', file: 'src/ops.cpp',
    sig: ['// C = A · B,   A: [n, k],  B: [k, m],  C: [n, m]', 'dA = dC · Bᵀ    // [n, m] · [m, k] = [n, k]', 'dB = Aᵀ · dC    // [k, n] · [n, m] = [k, m]'],
    what: 'Градиенты обоих сомножителей — тоже матричные умножения.',
    how: [
      'Вывод через элементы: `C[i][j] = Σ_p A[i][p]·B[p][j]`, значит `dC[i][j]/dA[i][p] = B[p][j]`. Собираем по j: `dA[i][p] = Σ_j dC[i][j]·B[p][j] = (dC · Bᵀ)[i][p]`.',
      'Проверка по формам: формы результата обязаны совпасть с формами A и B — так формулу легко вспомнить.',
      'Внутри backward используется `matmul_values` (без графа), и только для входов с `requires_grad`.',
    ],
  }),

  h2('Редукции: max и argmax'),
  ...entry({
    name: '`max` — backward',
    what: 'Градиент проходит только в элемент, который оказался максимумом: `mask * g`, где `mask = (a == max(a, dim, keepdim=true))`.',
    notes: ['Если несколько элементов равны максимуму, градиент получат все они (в PyTorch — один). Для обучения это не важно.'],
  }),
  ...entry({
    name: '`argmax(a, dim)`',
    sig: ['Tensor argmax(const Tensor& a, std::size_t dim);'],
    what: 'Номер максимального элемента вдоль оси (как float). Используется в `accuracy`: предсказанный класс = argmax логитов.',
    notes: ['Индексы не дифференцируемы — граф не строится.'],
  }),
  ...entry({
    name: '`DimSplit`, `split_at`, `reduced_shape` (внутренние)',
    sig: ['struct DimSplit { std::size_t outer, size, inner; };', 'DimSplit split_at(const Shape& shape, std::size_t dim);', 'Shape reduced_shape(Shape shape, std::size_t dim, bool keepdim);'],
    what: 'Общие кусочки `reduce_dim` и `argmax`: разбиение формы на [outer, size, inner] (с проверкой dim) и форма результата редукции.',
  }),

  h2('Softmax'),
  ...entry({
    name: '`softmax(a, dim)` и `log_softmax(a, dim)`',
    sig: ['Tensor softmax(const Tensor& a, std::size_t dim);       // e^x / Σ e^x', 'Tensor log_softmax(const Tensor& a, std::size_t dim);   // x - log Σ e^x'],
    what: 'Превращают «сырые» оценки классов (логиты) в вероятности (softmax) или их логарифмы (log_softmax).',
    how: [
      '**Численная стабильность:** e^1000 = бесконечность. Но softmax не меняется от вычитания константы: `softmax(x) = softmax(x − m)`. Берём m = max(x) — тогда наибольший показатель равен 0, и переполнения нет.',
      '`m` отсоединён (`detach()`): математически он на результат не влияет, поэтому и в градиент не должен вносить вклад.',
      'Собраны из `exp`, `log`, `sum`, `−`, `/` — backward выводится автоматически, отдельной формулы нет.',
    ],
    example: ['softmax([1, 2, 3])             = [0.090, 0.245, 0.665]', 'softmax([1000, 1000, 1000])    = [0.333, 0.333, 0.333]  // без переполнения'],
  }),

  h2('Backward для views'),
  table(['View', 'Обратная операция над g'], [
    ['`view(shape)` / `reshape`', '`g.reshape(старая форма)`'],
    ['`transpose(d0, d1)`', '`g.transpose(d0, d1)` — повторное транспонирование возвращает обратно'],
    ['`permute(dims)`', '`g.permute(inverse)`, где `inverse[dims[i]] = i`'],
    ['`expand(shape)`', '`sum_to_shape(g, старая форма)`'],
    ['`clone()`', '`g` без изменений'],
    ['`unsqueeze` / `squeeze`', 'отдельной формулы нет — они вызывают `reshape`'],
  ], [3000, 6600]),
  p('`contiguous()` либо возвращает тот же тензор, либо вызывает `clone()` — поэтому тоже не требует своей формулы.'),

  h2('Проверка градиентов: gradcheck'),
  ...entry({
    name: '`gradcheck(op, inputs, eps, tol)`', file: 'tests/test_autograd.cpp',
    sig: ['void gradcheck(const std::function<Tensor(const std::vector<Tensor>&)>& op,', '               std::vector<Tensor> inputs, float eps = 1e-2f, float tol = 2e-2f);'],
    what: 'Сравнивает градиент из `backward()` с **численной** производной по определению.',
    how: [
      'Функция для проверки: `f = sum(op(inputs) * w)`, где `w` — случайные веса. Без `w` проверялась бы только сумма градиентов, и ошибки «разного знака» могли бы взаимно погаситься.',
      'Аналитический градиент: `f().backward()`, затем `x.grad()`.',
      'Численный: для каждого элемента `(f(x + eps) − f(x − eps)) / (2·eps)` — центральная разность, ошибка O(eps²).',
      'Сравнение с относительным допуском: `|a − n| ≤ tol · max(1, |n|)`.',
    ],
    example: ['gradcheck([](auto& in) { return matmul(in[0], in[1]); },', '          {Tensor::randn({3, 4}), Tensor::randn({4, 2})});'],
    notes: [
      'Для `log` и дробных степеней входы положительные; для `relu` — подальше от нуля (в изломе производная не определена).',
      'Проверено «мутационным тестом»: если намеренно испортить формулу (`1 − t` вместо `1 − t²` в tanh, ×2 в sum), тесты падают.',
      'eps для float нельзя брать слишком маленьким: при 1e-6 разность f(x+eps) − f(x−eps) утонет в ошибках округления float (~7 цифр).',
    ],
  }),

  h2('Идеи C++ этого дня'),
  h3('Generic lambda'),
  p('`[](auto& in) { return exp(in[0]); }` — параметр типа `auto` (C++14). Такая лямбда — шаблон; при передаче в `std::function<Tensor(const std::vector<Tensor>&)>` компилятор подставляет нужный тип.'),
  h3('Циклы владения shared_ptr'),
  p('Если A держит shared_ptr на B, а B — на A, счётчики никогда не дойдут до нуля, и память утечёт. Правило autograd: ссылки идут только от выхода к входам. Поэтому лямбды backward захватывают входы, но никогда не выход.'),
  h3('Численные методы и float'),
  p('Центральная разность точнее односторонней (ошибка O(eps²) против O(eps)). Слишком маленький eps — ошибка округления, слишком большой — ошибка аппроксимации. Для float разумно eps ≈ 1e-2…1e-3.'),
];
