import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os

# Параметры из задания
sizes = (10, 100, 1000)
distributions = {
    "normal": {"name": "Нормальное N(0, 1)", "type": "continuous"},
    "cauchy": {"name": "Коши C(0, 1)", "type": "continuous"},
    "laplace": {"name": "Лапласа L(0, 1/sqrt(2))", "type": "continuous"},
    "poisson": {"name": "Пуассона P(10)", "type": "discrete"},
    "uniform": {"name": "Равномерное U(-sqrt(3), sqrt(3))", "type": "continuous"}
}

# Создаем папку для графиков, если её нет
if not os.path.exists("plots"):
    os.makedirs("plots")

for dist_key, dist_info in distributions.items():
    for n in sizes:
        filename = f"{dist_key}_{n}.csv"
        
        if not os.path.exists(filename):
            print(f"Файл {filename} не найден!")
            continue
            
        data = np.loadtxt(filename)
        plt.figure(figsize=(8, 5))
        
        # Обработка непрерывных распределений
        if dist_info["type"] == "continuous":
            bins_count = int(1 + np.log2(n))
            
            # СНАЧАЛА определяем границы графика
            x_min, x_max = np.min(data) - 1, np.max(data) + 1
            if dist_key == "cauchy":
                x_min, x_max = -10, 10 # Обрезаем выбросы Коши
                plt.xlim(x_min, x_max)
                
            # ТЕПЕРЬ рисуем гистограмму, добавив параметр range=(x_min, x_max)!
            plt.hist(data, bins=bins_count, density=True, range=(x_min, x_max), alpha=0.6, color='skyblue', edgecolor='black', label='Эмпирическая гистограмма')
            
            x = np.linspace(x_min, x_max, 1000)
            
            # Подбираем теоретическую функцию плотности (PDF)
            if dist_key == "normal":
                y = stats.norm.pdf(x, loc=0, scale=1)
            elif dist_key == "cauchy":
                y = stats.cauchy.pdf(x, loc=0, scale=1)
            elif dist_key == "laplace":
                y = stats.laplace.pdf(x, loc=0, scale=1/np.sqrt(2))
            elif dist_key == "uniform":
                y = stats.uniform.pdf(x, loc=-np.sqrt(3), scale=2*np.sqrt(3))
                
            plt.plot(x, y, 'red', linewidth=2, label='Теоретическая плотность (PDF)')
        # Обработка дискретного распределения (Пуассона)
        else: 
            # Для дискретных величин бины центрируются по целым числам
            bins_count = np.arange(np.min(data)-0.5, np.max(data)+1.5, 1)
            plt.hist(data, bins=bins_count, density=True, alpha=0.6, color='lightgreen', edgecolor='black', label='Эмпирическая гистограмма')
            
            x = np.arange(np.min(data), np.max(data)+1)
            # Для дискретных используем функцию вероятности (PMF), а не плотности
            y = stats.poisson.pmf(x, mu=10)
            plt.plot(x, y, 'ro', label='Теоретическая вероятность (PMF)')
            plt.vlines(x, 0, y, colors='r', lw=2, alpha=0.5)

        plt.title(f'{dist_info["name"]}, n={n}')
        plt.xlabel('Значение (x)')
        plt.ylabel('Плотность / Вероятность')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        
        # Сохраняем в папку plots
        plt.savefig(f"plots/{dist_key}_{n}.png")
        plt.close() # Очищаем память

print("Все 15 графиков успешно сохранены в папку 'plots'!")