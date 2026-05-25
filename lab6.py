import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Фиксируем seed для красивых и воспроизводимых результатов
np.random.seed(42)

# 1. Задаем значения x на отрезке [-1.8, 2] с шагом 0.2 (ровно 20 точек)
x = np.linspace(-1.8, 2.0, 20)

# Генерируем нормальный шум
eps = np.random.normal(0, 1, 20)

# Генерируем истинный отклик y = 2 + 2x + eps
y = 2 + 2 * x + eps

# Истинные параметры
a_true = 2.0
b_true = 2.0

# ---------------------------------------------------------
# ФУНКЦИИ И ВЫЧИСЛЕНИЯ
# ---------------------------------------------------------
# Целевая функция для Метод Наименьших Модулей (МНМ) - L1 норма
def lad_objective(params, x_val, y_val):
    a, b = params
    return np.sum(np.abs(y_val - (a + b * x_val)))

# Функция для расчета погрешностей
def calc_metrics(a_est, b_est):
    da = np.abs(a_true - a_est)
    db = np.abs(b_true - b_est)
    delta_a = (da / np.abs(a_true)) * 100
    delta_b = (db / np.abs(b_true)) * 100
    return da, delta_a, db, delta_b

# --- БЕЗ ВЫБРОСОВ ---
# МНК (через встроенный полифит 1-й степени)
b_mnk_1, a_mnk_1 = np.polyfit(x, y, 1)

# МНМ (минимизируем L1, в качестве начального приближения берем результаты МНК)
res_1 = minimize(lad_objective, [ a_mnk_1, b_mnk_1 ], args=(x, y), method='Nelder-Mead')
a_mnm_1, b_mnm_1 = res_1.x

# --- С ВЫБРОСАМИ ---
y_out = y.copy()
y_out[ 0 ] += 10   # y_1 <- y_1 + 10
y_out[ -1 ] -= 10  # y_20 <- y_20 - 10

# МНК с выбросами
b_mnk_2, a_mnk_2 = np.polyfit(x, y_out, 1)

# МНМ с выбросами
res_2 = minimize(lad_objective, [ a_mnk_2, b_mnk_2 ], args=(x, y_out), method='Nelder-Mead')
a_mnm_2, b_mnm_2 = res_2.x

# ---------------------------------------------------------
# ВЫВОД ТАБЛИЦ В ТЕРМИНАЛ
# ---------------------------------------------------------
def print_table(title, a_mnk, b_mnk, a_mnm, b_mnm):
    da_k, da_k_pct, db_k, db_k_pct = calc_metrics(a_mnk, b_mnk)
    da_m, da_m_pct, db_m, db_m_pct = calc_metrics(a_mnm, b_mnm)
    
    print(f"\n{title}")
    print("-" * 65)
    print(f"{'Метод':<5} | {'a':<6} | {'Δa':<6} | {'δa, %':<7} | {'b':<6} | {'Δb':<6} | {'δb, %':<7}")
    print("-" * 65)
    print(f"{'МНК':<5} | {a_mnk:.3f}  | {da_k:.3f}  | {da_k_pct:>7.3f} | {b_mnk:.3f}  | {db_k:.3f}  | {db_k_pct:>7.3f}")
    print(f"{'МНМ':<5} | {a_mnm:.3f}  | {da_m:.3f}  | {da_m_pct:>7.3f} | {b_mnm:.3f}  | {db_m:.3f}  | {db_m_pct:>7.3f}")
    print("-" * 65)

print_table("Таблица 1. Оценки коэффициентов регрессии БЕЗ ВЫБРОСОВ", a_mnk_1, b_mnk_1, a_mnm_1, b_mnm_1)
print_table("Таблица 2. Оценки коэффициентов регрессии С ВЫБРОСАМИ", a_mnk_2, b_mnk_2, a_mnm_2, b_mnm_2)

# ---------------------------------------------------------
# ПОСТРОЕНИЕ ГРАФИКОВ
# ---------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# График 1: Без выбросов
ax1.scatter(x, y, color='black', label='Данные', zorder=5)
ax1.plot(x, 2 + 2*x, 'g--', linewidth=2, label='Истинная прямая')
ax1.plot(x, a_mnk_1 + b_mnk_1*x, 'r-', linewidth=2, label='МНК')
ax1.plot(x, a_mnm_1 + b_mnm_1*x, 'b-', linewidth=2, label='МНМ')

ax1.set_title('Выборка без выбросов')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend()

# График 2: С выбросами
# Рисуем обычные точки (со 2-й по 19-ю) черным
ax2.scatter(x[ 1:-1 ], y_out[ 1:-1 ], color='black', label='Данные', zorder=5)
# Рисуем выбросы (1-ю и 20-ю точки) большим красным кружком
ax2.scatter([ x[ 0 ], x[ -1 ] ], [ y_out[ 0 ], y_out[ -1 ] ], color='red', s=100, label='Выбросы', zorder=6)

ax2.plot(x, 2 + 2*x, 'g--', linewidth=2, label='Истинная прямая')
ax2.plot(x, a_mnk_2 + b_mnk_2*x, 'r-', linewidth=2, label='МНК (сдвинулся)')
ax2.plot(x, a_mnm_2 + b_mnm_2*x, 'b-', linewidth=2, label='МНМ (робастный)')

ax2.set_title('Выборка с выбросами')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.legend()

plt.tight_layout()
plt.savefig('lab6_regression.png', dpi=300)
print("\nУспех! Картинка lab6_regression.png создана и сохранена.")