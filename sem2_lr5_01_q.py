import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange

# Параметры задачи
a = 1
b = 9
n = 8
dx = (b - a) / n

# Таблица значений функции
x = np.linspace(a, b, n + 1)
y = x * np.cbrt(1 + x)

# Промежуточные точки для оценки отклонения
xj = np.array([a + dx * (i + 0.5) for i in range(n)])
yj_exact = xj * np.cbrt(1 + xj)

# --- Задание 1: Аппроксимация полиномами ---
def fit_polynomial(x_data, y_data, degree):
    """Аппроксимация полиномом степени `degree` методом МНК"""
    A = np.vander(x_data, degree + 1, increasing=True)
    coeffs = np.linalg.lstsq(A, y_data, rcond=None)[0]
    return coeffs

def evaluate_polynomial(coeffs, x):
    """Вычисление значения полинома с коэффициентами `coeffs` в точке x"""
    return sum(c * x**i for i, c in enumerate(coeffs))

# Подбор полиномов 2-5 степени
degrees = [2, 3, 4, 5]
results = []

for deg in degrees:
    coeffs = fit_polynomial(x, y, deg)
    results.append({
        'degree': deg,
        'coeffs': coeffs,
        'max_error': max(abs(yj_exact[i] - evaluate_polynomial(coeffs, xj[i])) for i in range(len(xj)))
    })

# --- Визуализация ---
plt.figure(figsize=(12, 8))
plt.plot(x, y, 'o', label='Исходная таблица')
plt.plot(np.linspace(a, b, 100), np.array([evaluate_polynomial(results[0]['coeffs'], xi) for xi in np.linspace(a, b, 100)]), '-', label=f'Полином {results[0]["degree"]} степени')
plt.plot(np.linspace(a, b, 100), np.array([evaluate_polynomial(results[1]['coeffs'], xi) for xi in np.linspace(a, b, 100)]), '--', label=f'Полином {results[1]["degree"]} степени')
plt.plot(np.linspace(a, b, 100), np.array([evaluate_polynomial(results[2]['coeffs'], xi) for xi in np.linspace(a, b, 100)]), '-.', label=f'Полином {results[2]["degree"]} степени')
plt.plot(np.linspace(a, b, 100), np.array([evaluate_polynomial(results[3]['coeffs'], xi) for xi in np.linspace(a, b, 100)]), ':', label=f'Полином {results[3]["degree"]} степени')
plt.legend()
plt.title('Аппроксимация полиномами')
plt.grid(True)
plt.show()

# --- Задание 2: Интерполяция Лагранжа ---
poly_lagrange = lagrange(x, y)
yj_lagrange = poly_lagrange(xj)
lagrange_error = abs(yj_exact - yj_lagrange)

# --- Вывод результатов ---
print("Результаты аппроксимации:")
for res in results:
    print(f"Полином {res['degree']} степени:")
    print(f"  Коэффициенты: {res['coeffs']}")
    print(f"  Максимальное отклонение: {res['max_error']:.4f}")

print("\nРезультаты интерполяции Лагранжа:")
print(f"Значения в промежуточных точках: {yj_lagrange}")
print(f"Точные значения: {yj_exact}")
print(f"Погрешность: {lagrange_error}")
print(f"Средняя погрешность: {np.mean(lagrange_error):.4f}")

# --- Анализ наилучшей аппроксимации ---
best_result = min(results, key=lambda x: x['max_error'])
print(f"\nНаилучшая аппроксимация: Полином {best_result['degree']} степени")
print(f"Обоснование: минимальное максимальное отклонение ({best_result['max_error']:.4f})")