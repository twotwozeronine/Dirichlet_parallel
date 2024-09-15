import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from joblib import Parallel, delayed
import multiprocessing
import time

# Точное решение задачи для тестирования
def exact_solution(x, lambda_):
    return np.sin(lambda_ * x)

# Правая часть уравнения для точного решения
def source_term(x, lambda_):
    return lambda_**2 * np.sin(lambda_ * x)

# Метод конечных элементов с кусочно-линейными базисными функциями
def finite_element_method(N, lambda_):
    # Границы области
    a, b = 0, 2 * np.pi
    h = (b - a) / N  # шаг сетки
    x = np.linspace(a, b, N + 1)

    # Матрица жесткости и вектор правой части
    K = sp.lil_matrix((N+1, N+1))
    F = np.zeros(N+1)

    # Локальная матрица жесткости для кусочно-линейных функций
    K_el = (1 / h) * np.array([[1, -1], [-1, 1]])

    # Заполнение матрицы жесткости
    for i in range(N):
        K[i:i+2, i:i+2] += K_el

    # Правая часть: интегрируем функцию источника для каждого элемента
    for i in range(N):
        x_i = x[i]
        x_ip1 = x[i+1]
        # Трапецкой формулой на каждом элементе
        F[i:i+2] += (h / 2) * np.array([source_term(x_i, lambda_), source_term(x_ip1, lambda_)])

    # Применение граничных условий
    K[0, :] = 0
    K[-1, :] = 0
    K[0, 0] = 1
    K[-1, -1] = 1
    F[0] = 0
    F[-1] = 0

    # Решение системы уравнений методом сопряжённых градиентов
    u_h, _ = spla.cg(K.tocsr(), F, tol=1e-10)
    return x, u_h

# Функция для вычисления ошибки
def compute_error(N, lambda_):
    x, u_h = finite_element_method(N, lambda_)
    u_exact = exact_solution(x, lambda_)
    error = np.sqrt(np.sum((u_exact - u_h)**2) * (2 * np.pi) / N)
    return (lambda_, N, error)

# Функция для вычисления допустимой ошибки
def compute_allowable_error(N, lambda_):
    a, b = 0, 2 * np.pi
    h = (b - a) / N
    x = np.linspace(a, b, N + 1)
    f = source_term(x, lambda_)
    f_norm = np.sqrt(np.sum(f**2) * (b - a) / N)
    allowable_error = (h**2) * f_norm
    return allowable_error

# Численный эксперимент с распараллеливанием
def run_experiment_parallel(lambdas, N_values, n_jobs=-1):
    """
    Запуск численного эксперимента с использованием параллельных вычислений.

    Args:
        lambdas (list): Список значений параметра lambda.
        N_values (list): Список значений числа элементов N.
        n_jobs (int): Количество параллельных задач. По умолчанию -1, что означает использование всех доступных ядер.

    Returns:
        results (list): Список кортежей (lambda, N, error, allowable_error, status).
    """
    tasks = [(lambda_, N) for lambda_ in lambdas for N in N_values]
    
    start_time = time.time()
    
    # Выполнение вычислений в параллельных задачах
    parallel_errors = Parallel(n_jobs=n_jobs)(
        delayed(compute_error)(N, lambda_) for (lambda_, N) in tasks
    )
    
    parallel_time = time.time() - start_time
    print(f"Время вычислений с распараллеливанием: {parallel_time:.2f} секунд")

    # Вычисление допустимых ошибок и оценка статуса
    results = []
    for i, (lambda_, N, error) in enumerate(parallel_errors):
        allowable_error = compute_allowable_error(N, lambda_)
        status = "В пределах допустимого" if error <= allowable_error else "Превышает допустимую"
        results.append((lambda_, N, error, allowable_error, status))
    
    return results, parallel_time

# Численный эксперимент без распараллеливания для сравнения
def run_experiment_serial(lambdas, N_values):
    """
    Запуск численного эксперимента без использования параллельных вычислений.

    Args:
        lambdas (list): Список значений параметра lambda.
        N_values (list): Список значений числа элементов N.

    Returns:
        results (list): Список кортежей (lambda, N, error, allowable_error, status).
    """
    results = []
    start_time = time.time()
    
    for lambda_ in lambdas:
        print(f"Lambda = {lambda_}")
        for N in N_values:
            computed = compute_error(N, lambda_)
            error = computed[2]
            allowable_error = compute_allowable_error(N, lambda_)
            status = "В пределах допустимого" if error <= allowable_error else "Превышает допустимую"
            results.append((computed[0], computed[1], error, allowable_error, status))
            print(f"N = {N}, Полученная ошибка = {error:.6e}, Допустимая ошибка = {allowable_error:.6e}, Статус: {status}")
        print()
    
    serial_time = time.time() - start_time
    print(f"Время вычислений без распараллеливания: {serial_time:.2f} секунд")
    
    return results, serial_time

# Оценка эффективности распараллеливания
def evaluate_parallel_efficiency(lambdas, N_values):
    print("Запуск эксперимента без распараллеливания...")
    serial_results, serial_time = run_experiment_serial(lambdas, N_values)
    print(f"Общее время без распараллеливания: {serial_time:.2f} секунд\n")
    
    print("Запуск эксперимента с распараллеливанием...")
    parallel_results, parallel_time = run_experiment_parallel(lambdas, N_values)
    print(f"Общее время с распараллеливанием: {parallel_time:.2f} секунд\n")
    
    speedup = serial_time / parallel_time if parallel_time > 0 else float('inf')
    print(f"Ускорение при распараллеливании: {speedup:.2f}x")
    
    # Проверка корректности результатов
    discrepancies = []
    for s, p in zip(serial_results, parallel_results):
        # Проверяем, что lambda и N совпадают
        if s[0] != p[0] or s[1] != p[1]:
            discrepancies.append(f"Несоответствие параметров для lambda={s[0]}, N={s[1]}")
            continue
        
        # Проверяем только статус
        if s[4] != p[4]:
            discrepancy_info = (
                f"Несоответствие статуса для lambda={s[0]}, N={s[1]}:\n"
                f"  Серийный статус: {s[4]}, Параллельный статус: {p[4]}"
            )
            discrepancies.append(discrepancy_info)
    
    if discrepancies:
        print("\nОбнаружены несоответствия между серийным и параллельным выполнением:")
        for d in discrepancies:
            print(d)
        raise AssertionError("Несоответствие статусов между серийными и параллельными результатами.")
    else:
        print("Результаты параллельных вычислений совпадают с серийными по статусу.")
    
    return serial_results, parallel_results, speedup

# Основная функция для запуска эксперимента
def main():
    # Параметры эксперимента
    lambdas = [1, 10, 50, 100]
    N_values = [10, 100, 1000, 10000]
    
    try:
        # Оценка эффективности распараллеливания
        serial_results, parallel_results, speedup = evaluate_parallel_efficiency(lambdas, N_values)
    except AssertionError as e:
        print(f"Ошибка проверки: {e}")
        return
    
    # Вывод результатов параллельного эксперимента
    print("\nРезультаты параллельного эксперимента:")
    for result in parallel_results:
        lambda_, N, error, allowable_error, status = result
        print(f"Lambda = {lambda_}, N = {N}, Ошибка = {error:.6e}, "
              f"Допустимая ошибка = {allowable_error:.6e}, Статус: {status}")

# Запуск основной функции
if __name__ == "__main__":
    main()