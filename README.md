# Модули нейронных сетей на основе PyTorch

Этот проект содержит реализацию нейронных сетей с нуля с использованием NumPy, вдохновленную архитектурой PyTorch. Основная цель — создать гибкую и расширяемую структуру для глубокого обучения без использования высокоуровневых библиотек.

---

## Структура проекта

- **Module**: базовый класс для всех нейронных сетей и операций.
- **Sequential**: контейнер для последовательной сборки моделей.
- **Слои нейронной сети**:
  - *Linear*: полносвязный слой.
  - *Conv2d*: двумерный сверточный слой.
  - *BatchNormalization*: нормализация по мини-батчам.
  - *Dropout*: слой регуляризации.
  - *Активационные функции*:
    - ReLU
    - LeakyReLU
    - ELU
    - SoftPlus
    - SoftMax
    - LogSoftMax
  - *Пулинговые слои*:
    - MaxPool2d
  - *Прочие слои*:
    - Flatten: преобразование входных данных в одномерный массив.

- **Функции потерь**:
  - MSECriterion: среднеквадратичная ошибка.
  - ClassNLLCriterion: отрицательный логарифм правдоподобия.

- **Оптимизаторы**:
  - SGD с моментумом.
  - Adam.

---

## Установка

Склонируйте репозиторий:
```bash
git clone https://github.com/ARR0S/PyTorchImplementation.git
cd custom-pytorch-modules
```

Установите необходимые пакеты:
```bash
pip install numpy scipy
```

---

## 🚀 Использование

### Импорт модулей:
```python
from modules import Sequential, Linear, ReLU, MSECriterion, SGD
```

### Создание модели:
```python
model = Sequential()
model.add(Linear(10, 50))
model.add(ReLU())
model.add(Linear(50, 1))
```

### Настройка функции потерь и оптимизатора:
```python
criterion = MSECriterion()
optimizer = SGD(model.getParameters(), model.getGradParameters(), {'learning_rate': 0.01})
```

### Цикл обучения:
```python
for epoch in range(100):
    output = model.forward(input_data)
    loss = criterion.forward(output, target)
    gradOutput = criterion.backward(output, target)
    model.backward(input_data, gradOutput)
    optimizer.step()
    print(f"Эпоха {epoch+1}, Потеря: {loss}")
```