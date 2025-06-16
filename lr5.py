import numpy as np
import matplotlib.pyplot as plt


def f(x):
    """Исходная функция: y = x * (1 + x)^(1/3)"""
    return x * np.cbrt(1 + x)


a = 1.0
b = 9.0
n = 8

delta_x = (b - a) / n

# Создаем массив узлов x_i и вычисляем значения функции y_i в этих узлах
x_i = np.array([a + i * delta_x for i in range(n + 1)])
y_i = f(x_i)

# Создаем массив промежуточных точек x_j для оценки погрешности
x_j = np.array([a + delta_x * (j + 0.5) for j in range(n)])
# Вычисляем точные значения функции в промежуточных точках
y_j_exact = f(x_j)


print("Исходные данные")
print(f"Функция: y = x * (1 + x)^(1/3)")
print(f"Интервал: [{a}, {b}]")
print(f"Количество отрезков n = {n}, шаг delta_x = {delta_x:.4f}\n")

print("Таблица узловых точек (x_i, y_i):")
for i in range(len(x_i)):
    print(f"x_{i} = {x_i[i]:.4f}, y_{i} = {y_i[i]:.4f}")
print("-" * 30 + "\n")



print("Задание 1: Метод наименьших квадратов (МНК)\n")

# Словарь для хранения максимальных отклонений для каждого полинома
max_deviations_mnk = {}
# Словарь для хранения функций полиномов для последующего построения графиков
polynomials = {}

# Перебираем степени полиномов от 2 до 5
for m in range(2, 6):
    print(f"Аппроксимация полиномом степени {m}")

    # Формируем матрицу системы нормальных уравнений (метод Гаусса)
    # A[k][j] = sum(x_i ^ (k + j))
    A = np.zeros((m + 1, m + 1))
    for k in range(m + 1):
        for j in range(m + 1):
            A[k, j] = np.sum(x_i**(k + j))

    # Формируем вектор правых частей
    # B[k] = sum(y_i * x_i^k)
    B = np.zeros(m + 1)
    for k in range(m + 1):
        B[k] = np.sum(y_i * x_i**k)

    # Решаем систему линейных уравнений A*c = B для нахождения коэффициентов c
    try:
        coeffs = np.linalg.solve(A, B)
        # Сохраняем функцию полинома
        polynomials[m] = np.poly1d(np.flip(coeffs)) # np.poly1d принимает коэффициенты от старшей степени к младшей

        print("Коэффициенты полинома (от a0 до a_m):")
        for i, c in enumerate(coeffs):
            print(f"a_{i} = {c:.4f}")
        print()

        # Вычисляем значения аппроксимирующего полинома в узловых точках x_i
        y_approx_i = polynomials[m](x_i)

        print("Погрешности в узловых точках x_i:")
        print("i |   x_i  |  y_exact | y_approx |Абс Ошибка | Отн Ошибка (%)")
        print("-" * 65)
        for i in range(len(x_i)):
            abs_err = np.abs(y_i[i] - y_approx_i[i])
            rel_err = (abs_err / np.abs(y_i[i])) * 100 if y_i[i] != 0 else 0
            print(f"{i:1d} | {x_i[i]:6.3f} | {y_i[i]:8.4f} | {y_approx_i[i]:8.4f} | {abs_err:9.4f} | {rel_err:10.2f}%")
        print()

        # Вычисляем значения в промежуточных точках x_j
        y_approx_j = polynomials[m](x_j)
        
        # Находим максимальное отклонение в промежуточных точках
        max_dev = np.max(np.abs(y_j_exact - y_approx_j))
        max_deviations_mnk[m] = max_dev
        print(f"Максимальное отклонение в промежуточных точках x_j: {max_dev:.4f}\n")

    except np.linalg.LinAlgError:
        print(f"Не удалось решить систему для полинома степени {m}. Матрица вырождена.\n")

# Обоснование выбора наилучшей аппроксимирующей функции
best_degree = min(max_deviations_mnk, key=max_deviations_mnk.get)
print("--- Выбор наилучшей аппроксимирующей функции (МНК) ---")
print("Для выбора наилучшего полинома сравним максимальные отклонения в промежуточных точках.")
print("Чем меньше это отклонение, тем лучше полином описывает поведение функции МЕЖДУ узлами.")
for degree, dev in max_deviations_mnk.items():
    print(f"Степень {degree}: Макс. отклонение = {dev:.4f}")
print(f"\nНаилучшей является аппроксимация полиномом степени {best_degree},")
print(f"так как он имеет минимальное максимальное отклонение ({max_deviations_mnk[best_degree]:.4f}).")
print("-" * 30 + "\n")


print("Задание 2: Интерполяционный многочлен Лагранжа\n")

def lagrange_interpolation(x_nodes, y_nodes, x_eval):
    """
    Вычисляет значение интерполяционного многочлена Лагранжа.
    x_nodes: узлы интерполяции (массив)
    y_nodes: значения в узлах (массив)
    x_eval: точка, в которой вычисляется значение
    """
    n_nodes = len(x_nodes)
    result = 0.0
    for j in range(n_nodes):
        # Вычисляем базисный полином l_j(x)
        l_j = 1.0
        for i in range(n_nodes):
            if i != j:
                l_j *= (x_eval - x_nodes[i]) / (x_nodes[j] - x_nodes[i])
        result += y_nodes[j] * l_j
    return result

# Вычисляем значения с помощью многочлена Лагранжа в промежуточных точках x_j
y_lagrange_j = np.array([lagrange_interpolation(x_i, y_i, x) for x in x_j])

print("Сравнение точных и вычисленных (Лагранж) значений в промежуточных точках x_j:")
print("j |   x_j  |  y_exact | y_lagrange |Абс Ошибка | Отн Ошибка (%)")
print("-" * 68)
for j in range(len(x_j)):
    abs_err = np.abs(y_j_exact[j] - y_lagrange_j[j])
    rel_err = (abs_err / np.abs(y_j_exact[j])) * 100 if y_j_exact[j] != 0 else 0
    print(f"{j:1d} | {x_j[j]:6.3f} | {y_j_exact[j]:8.4f} | {y_lagrange_j[j]:10.4f} | {abs_err:9.4f} | {rel_err:10.4f}%")
print("-" * 30 + "\n")



print("Построение графиков.")

x_smooth = np.linspace(a, b, 200)
y_smooth = f(x_smooth)

plt.figure(figsize=(14, 8))

# График 1: Исходная функция и МНК
plt.subplot(1, 2, 1)
plt.plot(x_smooth, y_smooth, 'k-', label='Исходная функция f(x)', linewidth=2.5)
plt.plot(x_i, y_i, 'ko', label='Узловые точки (x_i, y_i)', markersize=8)
colors = ['r', 'g', 'b', 'c']
for i, (degree, poly) in enumerate(polynomials.items()):
    plt.plot(x_smooth, poly(x_smooth), linestyle='--', color=colors[i], label=f'МНК, степень {degree}')
plt.title('Аппроксимация методом наименьших квадратов')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.minorticks_on()

# График 2: Исходная функция и интерполяция Лагранжа
plt.subplot(1, 2, 2)
plt.plot(x_smooth, y_smooth, 'k-', label='Исходная функция f(x)', linewidth=2.5)
plt.plot(x_i, y_i, 'ko', label='Узловые точки (x_i, y_i)', markersize=8)
plt.plot(x_j, y_lagrange_j, 'm^', label='Точки, вычисленные по Лагранжу', markersize=8)
plt.plot(x_j, y_j_exact, 'bx', label='Точные значения в точках x_j', markersize=8, mew=2)
plt.title('Интерполяция многочленом Лагранжа')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.minorticks_on()

plt.suptitle('Результаты аппроксимации и интерполяции', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
