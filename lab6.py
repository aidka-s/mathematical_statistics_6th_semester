import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Мы убрали фиксацию seed. Теперь каждый запуск генерирует честную случайную выборку!

# 1. Задаем значения x на отрезке [-1.8, 2] с шагом 0.2 (20 точек)
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
def lad_objective(params, x_val, y_val):
    a, b = params
    return np.sum(np.abs(y_val - (a + b * x_val)))

def calc_metrics(a_est, b_est):
    da = np.abs(a_true - a_est)
    db = np.abs(b_true - b_est)
    delta_a = (da / np.abs(a_true)) * 100
    delta_b = (db / np.abs(b_true)) * 100
    return da, delta_a, db, delta_b

# --- БЕЗ ВЫБРОСОВ ---
b_mnk_1, a_mnk_1 = np.polyfit(x, y, 1)
res_1 = minimize(lad_objective, [ a_mnk_1, b_mnk_1 ], args=(x, y), method='Nelder-Mead')
a_mnm_1, b_mnm_1 = res_1.x

# --- С ВЫБРОСАМИ ---
y_out = y.copy()
y_out[ 0 ] += 10
y_out[ -1 ] -= 10

b_mnk_2, a_mnk_2 = np.polyfit(x, y_out, 1)
res_2 = minimize(lad_objective, [ a_mnk_2, b_mnk_2 ], args=(x, y_out), method='Nelder-Mead')
a_mnm_2, b_mnm_2 = res_2.x

# ---------------------------------------------------------
# ВЫВОД ТАБЛИЦ
# ---------------------------------------------------------
def print_table(title, a_mnk, b_mnk, a_mnm, b_mnm):
    da_k, da_k_pct, db_k, db_k_pct = calc_metrics(a_mnk, b_mnk)
    da_m, da_m_pct, db_m, db_m_pct = calc_metrics(a_mnm, b_mnm)
    
    print(f"\n{title}")
    print("-" * 65)
    print(f"{'Метод':<5} | {'a':<6} | {'Δa':<6} | {'δa, %':<7} | {'b':<6} | {'Δb':<6} | {'δb, %':<7}")
    print("-" * 65)
    print(f"{'МНК':<5} | {a_mnk:.4f} | {da_k:.4f} | {da_k_pct:>7.2f} | {b_mnk:.4f} | {db_k:.4f} | {db_k_pct:>7.2f}")
    print(f"{'МНМ':<5} | {a_mnm:.4f} | {da_m:.4f} | {da_m_pct:>7.2f} | {b_mnm:.4f} | {db_m:.4f} | {db_m_pct:>7.2f}")
    print("-" * 65)

print_table("Таблица 1. Оценки коэффициентов регрессии БЕЗ ВЫБРОСОВ", a_mnk_1, b_mnk_1, a_mnm_1, b_mnm_1)
print_table("Таблица 2. Оценки коэффициентов регрессии С ВЫБРОСАМИ", a_mnk_2, b_mnk_2, a_mnm_2, b_mnm_2)

# ---------------------------------------------------------
# ПОСТРОЕНИЕ ГРАФИКОВ (в стиле второго отчета)
# ---------------------------------------------------------
# График 1: Без выбросов
plt.figure(figsize=(8, 6))
plt.scatter(x, y, color='tab:blue', zorder=5)
plt.plot(x, a_mnk_1 + b_mnk_1*x, label='МНК', color='tab:blue', linewidth=2)
plt.plot(x, a_mnm_1 + b_mnm_1*x, label='МНМ', color='tab:orange', linewidth=2)
plt.grid(True, linestyle='-', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig('lab6_no_outliers.png', dpi=300)
plt.close()

# График 2: С выбросами
plt.figure(figsize=(8, 6))
plt.scatter(x, y_out, color='tab:blue', zorder=5)
plt.plot(x, a_mnk_2 + b_mnk_2*x, label='МНК', color='tab:blue', linewidth=2)
plt.plot(x, a_mnm_2 + b_mnm_2*x, label='МНМ', color='tab:orange', linewidth=2)
plt.grid(True, linestyle='-', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig('lab6_outliers.png', dpi=300)
plt.close()

print("\nУспех! Сохранены два файла: lab6_no_outliers.png и lab6_outliers.png")