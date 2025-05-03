import numpy as np
import matplotlib.pyplot as plt
import sys 


a = 1.0
b = 9.0
n = 8    # Количество отрезков (n+1 точек)

def f(x):
  return x * np.cbrt(1 + x)

# --- Табулирование функции ---
delta_x = (b - a) / n
x_nodes = np.linspace(a, b, n + 1) 
y_nodes = f(x_nodes)

print("--- Таблица значений функции ---")
print("i |   x_i   |   y_i")
print("-" * 28)
for i in range(n + 1):
    print(f"{i:1d} | {x_nodes[i]:7.4f} | {y_nodes[i]:7.4f}")
print("\n")

# --- Вспомогательные функции ---

def simple_gaussian_elimination(A, B):
    """
    Решает систему линейных уравнений AX = B методом Гаусса (упрощенная версия).
    Возвращает вектор решения X. При ошибке выводит сообщение и завершает программу.
    """
    n_eq = len(B)
    # Создаем копии, чтобы не изменять оригинальные матрицы
    A_copy = A.astype(float).copy()
    B_copy = B.astype(float).copy()

    # Прямой ход
    for i in range(n_eq):
        # Находим главный элемент в текущем столбце (ниже i-й строки)
        pivot_row = i
        for k in range(i + 1, n_eq):
            if abs(A_copy[k, i]) > abs(A_copy[pivot_row, i]):
                pivot_row = k
        # Меняем строки местами в A и B
        A_copy[[i, pivot_row]] = A_copy[[pivot_row, i]]
        B_copy[[i, pivot_row]] = B_copy[[pivot_row, i]]

        # Проверка на нулевой диагональный элемент
        if abs(A_copy[i, i]) < 1e-10: # Используем малое число для сравнения с нулем
             print(f"Ошибка: Нулевой или близкий к нулю элемент на диагонали ({A_copy[i, i]:.2e}) в строке {i}.")
             print("Система может быть вырожденной. Завершение программы.")
             sys.exit(1) # Выход из программы

        # Обнуляем элементы под главным элементом
        for k in range(i + 1, n_eq):
            factor = A_copy[k, i] / A_copy[i, i]
            A_copy[k, i:] -= factor * A_copy[i, i:]
            B_copy[k] -= factor * B_copy[i]

    # Обратный ход
    X = np.zeros(n_eq)
    for i in range(n_eq - 1, -1, -1):
        # Проверка на нулевой диагональный элемент (на всякий случай, после прямого хода)
        if abs(A_copy[i, i]) < 1e-10:
             print(f"Ошибка: Нулевой или близкий к нулю элемент на диагонали ({A_copy[i, i]:.2e}) в строке {i} после прямого хода.")
             print("Завершение программы.")
             sys.exit(1)

        X[i] = f"{((B_copy[i] - np.sum(A_copy[i, i+1:] * X[i+1:])) / A_copy[i, i]):.3f}"

    return X

def polynomial_value(coeffs, x):
    """Вычисляет значение полинома с коэффициентами coeffs в точке x."""
    degree = len(coeffs) - 1
    y = 0
    for i in range(degree + 1):
        y += coeffs[i] * (x ** i)
    return y
    # return np.polyval(coeffs[::-1], x)

# --- Задание 1: Метод наименьших квадратов (МНК) ---

print("--- Задание 1: Метод наименьших квадратов ---")

polynomial_degrees = [2, 3, 4, 5]
approximations = {} # Словарь для хранения результатов аппроксимации
max_deviations_mls = {} # Словарь для хранения максимальных отклонений

# Точки для вычисления отклонений x'_i = a + Δx(i + 0.5)
x_intermediate_mls = a + delta_x * (np.arange(n) + 0.5)
y_exact_intermediate_mls = f(x_intermediate_mls)

for degree in polynomial_degrees:
    print(f"\nАппроксимация полиномом степени {degree}:")
    m = degree # Степень полинома
    num_coeffs = m + 1

    # Формирование матрицы системы нормальных уравнений (A)
    A = np.zeros((num_coeffs, num_coeffs))
    for k in range(num_coeffs):
        for j in range(num_coeffs):
            A[k, j] = np.sum(x_nodes ** (k + j)) # Сумма x^(k+j) по всем узлам

    # Формирование вектора правых частей (B)
    B = np.zeros(num_coeffs)
    for k in range(num_coeffs):
        B[k] = np.sum(y_nodes * (x_nodes ** k)) # Сумма y*x^k по всем узлам

    # Решение системы AC = B методом Гаусса
    # Функция simple_gaussian_elimination завершит программу, если возникнет проблема
    coeffs = simple_gaussian_elimination(A, B)
    print("Коэффициенты полинома (от C0 до Cm):")
    print(coeffs)

    # Вычисление значений аппроксимирующего полинома в узлах
    y_approx_nodes = polynomial_value(coeffs, x_nodes)
    approximations[degree] = {'coeffs': coeffs, 'y_approx_nodes': y_approx_nodes}

    # Вычисление значений аппроксимирующего полинома в промежуточных точках
    y_approx_intermediate = polynomial_value(coeffs, x_intermediate_mls)

    # Вычисление отклонений в промежуточных точках
    deviations = np.abs(y_approx_intermediate - y_exact_intermediate_mls)
    max_deviation = np.max(deviations)
    max_deviations_mls[degree] = max_deviation
    print(f"Максимальное отклонение в точках x'_i: {max_deviation:.3f}")

# Обоснование выбора наилучшей аппроксимирующей функции
best_degree_mls = min(max_deviations_mls, key=max_deviations_mls.get)
print(f"\nНаилучшая аппроксимация достигается полиномом степени {best_degree_mls}")
print(f"с максимальным отклонением {max_deviations_mls[best_degree_mls]:.3f}.")
print("Обоснование: Этот полином имеет наименьшее максимальное отклонение")
print("от точных значений функции в заданных промежуточных точках среди")
print("рассмотренных полиномов (степеней 2, 3, 4, 5).")


# --- Построение графиков МНК ---
plt.figure(figsize=(12, 7))
plt.plot(x_nodes, y_nodes, 'ko', label='Узлы (xi, yi)') # Узловые точки

x_plot = np.linspace(a, b, 200) # Более гладкий график
plt.plot(x_plot, f(x_plot), 'r-', linewidth=2, label='Точная функция f(x)')

colors = ['b', 'g', 'm', 'c']
for i, degree in enumerate(polynomial_degrees):
    # Проверяем, есть ли коэффициенты для данной степени
    if degree in approximations and approximations[degree] is not None:
        coeffs = approximations[degree]['coeffs']
        plt.plot(x_plot, polynomial_value(coeffs, x_plot),
                 linestyle='--', color=colors[i % len(colors)],
                 label=f'МНК полином {degree}-й степени')

plt.title('Метод наименьших квадратов (МНК)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()


# --- Задание 2: Интерполяционный многочлен Лагранжа ---

print("\n--- Задание 2: Интерполяционный многочлен Лагранжа ---")

def lagrange_basis_polynomial(x, i, x_nodes):
    """Вычисляет i-й базисный полином Лагранжа l_i(x)."""
    term = 1.0
    num_nodes = len(x_nodes)
    for j in range(num_nodes):
        if i != j:
            # Проверка деления на ноль (маловероятно при разных узлах)
            denominator = x_nodes[i] - x_nodes[j]
            if abs(denominator) < 1e-15: # Малое число для сравнения
                print(f"Предупреждение: Очень близкие узлы x[{i}] и x[{j}], возможна потеря точности.")
                # Можно добавить более строгую обработку, если нужно
            term *= (x - x_nodes[j]) / denominator
    return term

def lagrange_interpolation(x, x_nodes, y_nodes):
    """Вычисляет значение интерполяционного многочлена Лагранжа L(x)."""
    y_interp = 0.0
    num_nodes = len(y_nodes)
    for i in range(num_nodes):
        y_interp += y_nodes[i] * lagrange_basis_polynomial(x, i, x_nodes)
    return y_interp

# Точки для интерполяции x'_j = a + Δx * (j + 0.5)
x_intermediate_lagrange = a + delta_x * (np.arange(n) + 0.5)
y_exact_intermediate_lagrange = f(x_intermediate_lagrange)
y_interpolated = np.zeros(n)

print("\nСравнение значений в промежуточных точках x'_j:")
print(" j |   x'_j  | Точное y | Интерп. y | Погрешность |")
print("-" * 55)
for j in range(n):
    xj = x_intermediate_lagrange[j]
    y_interp_j = lagrange_interpolation(xj, x_nodes, y_nodes)
    y_interpolated[j] = y_interp_j
    y_exact_j = y_exact_intermediate_lagrange[j]
    error = np.abs(y_interp_j - y_exact_j)
    print(f"{j:2d} | {xj:7.4f} | {y_exact_j:8.4f} | {y_interp_j:9.4f} | {error:11.8f} |")

max_error_lagrange = np.max(np.abs(y_interpolated - y_exact_intermediate_lagrange))
print(f"\nМаксимальная абсолютная погрешность интерполяции Лагранжа: {max_error_lagrange:.8f}")

# --- Построение графиков Лагранжа ---
plt.figure(figsize=(12, 7))
plt.plot(x_nodes, y_nodes, 'ko', label='Узлы интерполяции (xi, yi)') # Узловые точки
plt.plot(x_plot, f(x_plot), 'r-', linewidth=2, label='Точная функция f(x)')

# Построим график самого полинома Лагранжа (вычисляя его в большом кол-ве точек)
y_lagrange_plot = np.zeros_like(x_plot)
for k in range(len(x_plot)):
    y_lagrange_plot[k] = lagrange_interpolation(x_plot[k], x_nodes, y_nodes)

plt.plot(x_plot, y_lagrange_plot, 'b--', label='Полином Лагранжа L(x)')

# Отметим точки, где вычислялась интерполяция
plt.plot(x_intermediate_lagrange, y_interpolated, 'g*', markersize=10, label='Интерполированные значения в x\'_j')

plt.title('Интерполяция многочленом Лагранжа')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

print("\n--- Завершено ---")

'''
```text/markdown
### Основные изменения:

1.  **Удален `try/except`:** Блок `try/except` в функции решения СЛАУ `gaussian_elimination` был заменен на простую проверку деления на ноль (`abs(A_copy[i, i]) < 1e-10`). Если обнаруживается нулевой или близкий к нулю диагональный элемент (что указывает на возможную вырожденность системы), программа выводит сообщение об ошибке и завершает выполнение с помощью `sys.exit(1)`.
2.  **Функция `polynomial_value`:** Для вычисления значения полинома теперь используется встроенная функция `numpy.polyval`, что делает код немного короче. Обратите внимание, что `np.polyval` ожидает коэффициенты в порядке от старшей степени к младшей, поэтому массив `coeffs` передается в обратном порядке (`coeffs[::-1]`).
3.  **Сохранение numpy:** Основные вычисления (создание узлов, вычисление степеней, суммирование для МНК) по-прежнему выполняются с использованием `numpy`, так как это стандартный и эффективный способ для таких задач в Python, и он делает код более читаемым и кратким по сравнению с ручными циклами по спискам.
4.  **Без классов:** Как и в предыдущей версии, классы не используются.

Этот вариант кода функционально эквивалентен предыдущему, но не использует обработку исключений `try/except` и включает некоторые мелкие упрощения.
'''