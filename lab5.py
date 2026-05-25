import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

# Фиксируем seed для воспроизводимости графиков и таблиц
np.random.seed(42)

# Параметры задания
sample_sizes = [ 20, 60, 100 ]
rho_values = [ 0.0, 0.5, 0.9 ]

# Функция для вычисления квадрантного коэффициента корреляции
def quadrant_correlation(x, y):
    med_x = np.median(x)
    med_y = np.median(y)
    # np.sign возвращает -1, 0 или 1
    return np.mean(np.sign(x - med_x) * np.sign(y - med_y))

# Функция генерации нормального двумерного распределения
def get_normal_sample(n, rho):
    cov_matrix = [ [ 1.0, rho ], [ rho, 1.0 ] ]
    # Генерируем массив n x 2, затем транспонируем, чтобы получить отдельно x и y
    return np.random.multivariate_normal([ 0.0, 0.0 ], cov_matrix, n).T

# Функция генерации смеси нормальных распределений
def get_mixture_sample(n):
    # Основное распределение: 90%, N(0, 0, 1, 1, 0.9)
    cov1 = [ [ 1.0, 0.9 ], [ 0.9, 1.0 ] ]
    # Выбросы: 10%, N(0, 0, 10, 10, -0.9). Ковариация = rho * std1 * std2 = -0.9 * 10 * 10 = -90
    cov2 = [ [ 100.0, -90.0 ], [ -90.0, 100.0 ] ]
    
    n1 = int(0.9 * n)
    n2 = n - n1
    
    sample1 = np.random.multivariate_normal([ 0.0, 0.0 ], cov1, n1)
    if n2 > 0:
        sample2 = np.random.multivariate_normal([ 0.0, 0.0 ], cov2, n2)
        # Объединяем основную выборку и выбросы
        sample = np.vstack((sample1, sample2))
    else:
        sample = sample1
        
    np.random.shuffle(sample)
    return sample.T

# Функция для отрисовки эллипса рассеяния
def draw_ellipse(x, y, ax):
    cov = np.cov(x, y)
    pearson = cov[ 0, 1 ] / np.sqrt(cov[ 0, 0 ] * cov[ 1, 1 ])
    
    # Находим собственные значения и векторы ковариационной матрицы
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    
    # Сортируем их по убыванию
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[ order ]
    eigenvectors = eigenvectors[ :, order ]
    
    # Угол наклона (в градусах)
    angle = np.degrees(np.arctan2(*eigenvectors[ :, 0 ][::-1]))
    
    # Ширина и высота эллипса для 2 стандартных отклонений (охватывает ~95% точек)
    width = 4 * np.sqrt(eigenvalues[ 0 ]) 
    height = 4 * np.sqrt(eigenvalues[ 1 ])
    
    ell = Ellipse(xy=(np.mean(x), np.mean(y)), width=width, height=height, 
                  angle=angle, edgecolor='red', fc='None', lw=2)
    ax.add_patch(ell)

# ---------------------------------------------------------
# ЧАСТЬ 1: Симуляция 1000 итераций для числовых характеристик
# ---------------------------------------------------------
print("Расчет статистик (1000 итераций)... Пожалуйста, подождите пару секунд.")
print("-" * 80)
print(f"{'n':<4} | {'Распределение':<12} | {'E(Пирсон)':<9} | {'D(Пирсон)':<9} | {'E(Спирмен)':<10} | {'D(Спирмен)':<10} | {'E(Квадр)':<8} | {'D(Квадр)':<8}")
print("-" * 80)

# Список всех типов распределений для цикла
distributions = [ ("Норм, rho=0.0", 0.0), ("Норм, rho=0.5", 0.5), 
                  ("Норм, rho=0.9", 0.9), ("Смесь", "mix") ]

for n in sample_sizes:
    for name, rho in distributions:
        pear_list, spear_list, quad_list = [], [], []
        
        for _ in range(1000):
            if rho == "mix":
                x, y = get_mixture_sample(n)
            else:
                x, y = get_normal_sample(n, float(rho))
                
            pear_list.append(stats.pearsonr(x, y)[ 0 ])
            spear_list.append(stats.spearmanr(x, y)[ 0 ])
            quad_list.append(quadrant_correlation(x, y))
            
        print(f"{n:<4} | {name:<12} | {np.mean(pear_list):>9.3f} | {np.var(pear_list):>9.3f} | {np.mean(spear_list):>10.3f} | {np.var(spear_list):>10.3f} | {np.mean(quad_list):>8.3f} | {np.var(quad_list):>8.3f}")

print("-" * 80)

# ---------------------------------------------------------
# ЧАСТЬ 2: Построение графиков с эллипсами
# ---------------------------------------------------------
fig, axes = plt.subplots(3, 4, figsize=(20, 15))
fig.suptitle('Диаграммы рассеяния и эллипсы равновероятности', fontsize=20)

for row_idx, n in enumerate(sample_sizes):
    for col_idx, (name, rho) in enumerate(distributions):
        ax = axes[ row_idx, col_idx ]
        
        # Генерируем 1 выборку для графика
        if rho == "mix":
            x, y = get_mixture_sample(n)
            r_val = stats.pearsonr(x, y)[ 0 ]
            title = f"{name}, n={n}\nr = {r_val:.2f}"
        else:
            x, y = get_normal_sample(n, float(rho))
            r_val = stats.pearsonr(x, y)[ 0 ]
            title = f"Норм, n={n}, $\\rho$={rho}\nr = {r_val:.2f}"
            
        # Рисуем точки
        ax.scatter(x, y, s=15, alpha=0.6, color='gray')
        
        # Накладываем эллипс
        draw_ellipse(x, y, ax)
        
        # Добавляем оси X и Y, проходящие через нуль
        ax.axhline(0, color='black', linewidth=0.5, alpha=0.5)
        ax.axvline(0, color='black', linewidth=0.5, alpha=0.5)
        
        ax.set_title(title)
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Выравниваем масштаб осей для честного круга
        if rho != "mix":
            ax.set_xlim(-4, 4)
            ax.set_ylim(-4, 4)
            ax.set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.savefig('lab5_ellipses.png', dpi=300)
print("Успех! Картинка lab5_ellipses.png создана и сохранена.")