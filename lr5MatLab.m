clear;
clc;
close all;

f = @(x) x .* (1 + x).^(1/3);

a = 1;
b = 9;
n = 8;

delta_x = (b - a) / n;

% Создание вектора узловых точек x_i и вычисление значений y_i
x_i = a:delta_x:b;
y_i = f(x_i);

% Создание вектора промежуточных точек x_j для оценки погрешности
x_j = a + delta_x * (0.5 : 1 : n-0.5);
% Вычисление точных значений функции в промежуточных точках
y_j_exact = f(x_j);

fprintf('Исходные данные\n');
fprintf('Функция: y = x * (1 + x)^(1/3)\n');
fprintf('Интервал: [%d, %d]\n', a, b);
fprintf('Количество отрезков n = %d, шаг delta_x = %.4f\n\n', n, delta_x);

fprintf('Таблица узловых точек (x_i, y_i):\n');
for i = 1:length(x_i)
    fprintf('x_%d = %.4f, y_%d = %.4f\n', i-1, x_i(i), i-1, y_i(i));
end
fprintf('----------------------------------------\n\n');

fprintf('Задание 1: Метод наименьших квадратов (МНК)\n\n');

% Массив для хранения максимальных отклонений для каждого полинома
max_deviations_mnk = [];

figure('Name', 'Аппроксимация МНК', 'NumberTitle', 'off');
hold on;

x_smooth = linspace(a, b, 200);
y_smooth = f(x_smooth);
plot(x_smooth, y_smooth, 'k', 'LineWidth', 2.5, 'DisplayName', 'Исходная функция f(x)');

plot(x_i, y_i, 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'k', 'DisplayName', 'Узловые точки');

colors = ['r', 'g', 'b', 'm'];

for m = 2:5
    fprintf('Аппроксимация полиномом степени %d\n', m);
    
    % polyfit - встроенная функция MATLAB для аппроксимации полиномом
    % по методу наименьших квадратов. Она возвращает коэффициенты полинома.
    coeffs = polyfit(x_i, y_i, m);
    
    fprintf('Коэффициенты полинома:\n');
    for k = 1:length(coeffs)
        % Индекс коэффициента 'a' идет от старшей степени 'm' к младшей '0'
        fprintf('  a_%d = %+.4f\n', m-k+1, coeffs(k));
    end
    
    % polyval - вычисляет значения полинома с коэффициентами 'coeffs' в точках 'x_i'
    y_approx_i = polyval(coeffs, x_i);
    
    abs_err = abs(y_i - y_approx_i);
    rel_err = (abs_err ./ abs(y_i)) * 100;
    
    fprintf('\nПогрешности в узловых точках x_i:\n');
    fprintf(' i |   x_i  |  y_exact | y_approx |Абс Ошибка | Отн Ошибка(%%)\n');
    fprintf('----------------------------------------------------------------\n');
    for i = 1:length(x_i)
        fprintf('%2d | %6.3f | %8.4f | %8.4f | %9.4f | %11.4f%%\n', ...
                i-1, x_i(i), y_i(i), y_approx_i(i), abs_err(i), rel_err(i));
    end
    fprintf('\n');
    
    % Вычисление значений в промежуточных точках x_j
    y_approx_j = polyval(coeffs, x_j);
    
    % Нахождение и сохранение максимального отклонения
    max_dev = max(abs(y_j_exact - y_approx_j));
    max_deviations_mnk(end+1) = max_dev; % Добавляем значение в конец массива
    fprintf('Максимальное отклонение в промежуточных точках x_j: %.6f\n\n', max_dev);

    plot(x_smooth, polyval(coeffs, x_smooth), '--', 'Color', colors(m-1), 'LineWidth', 1.5, 'DisplayName', ['МНК, степень ' num2str(m)]);
end

% Находим минимальное отклонение и его индекс
[min_dev, best_idx] = min(max_deviations_mnk);
best_degree = best_idx + 1; % +1 потому что степени у нас 2, 3, 4, 5, а индексы 1, 2, 3, 4

fprintf('Выбор наилучшей аппроксимирующей функции (МНК)\n');
fprintf('Сравним максимальные отклонения в промежуточных точках:\n');
for i = 1:length(max_deviations_mnk)
    fprintf('Степень %d: Макс. отклонение = %.6f\n', i+1, max_deviations_mnk(i));
end

fprintf('\nНаилучшей является аппроксимация полиномом степени %d,\n', best_degree);
fprintf('так как он имеет минимальное максимальное отклонение (%.6f).\n', min_dev);
fprintf('------------------------------------------------------------\n\n');

title('Аппроксимация функции методом наименьших квадратов');
xlabel('x');
ylabel('y');
legend('show', 'Location', 'northwest');
