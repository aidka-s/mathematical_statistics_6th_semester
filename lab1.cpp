#include <iostream>
#include <fstream>
#include <vector>
#include <random> //тут лежат генераторы случайных чисел
#include <cmath>
#include <string>
#include <clocale>

using namespace std;

double generateLaplace(mt19937& gen) { //Функция распределения псевдослучайных чисел. mt19937 это механизм типа "Вихрь Мерсенна"
    double b = 1.0 / sqrt(2.0); //Параметр распределения по тз
    
    uniform_real_distribution<double> unif(0.0, 1.0); // Генерируем равномерное случайное число от 0 до 1
    double u = unif(gen);

    if (u < 0.5) { //задали формулу обратного преобразования для Лапласа, тк в cpp явно функцией не задашь
        return b * log(2.0 * u);
    } else {
        return -b * log(2.0 - 2.0 * u);
    }
}

// Пропишем функцию для сохранения массива в файл
void saveToFile(const vector<double>& data, const string& filename) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "A mistake appeared while generating a file" << filename << endl;
        return;
    }
    for (double val : data) {
        file << val << "\n";
    }
    file.close();
}



// Прописываем главную функцию, она будет генерировать массивы
int main() {
    setlocale(LC_ALL, "Russian");
    random_device rd;
    mt19937 gen(rd());

    vector<int> sizes = {10, 100, 1000};

    // Задаем стандартные распределения сразу с параметрами
    normal_distribution<double> normal_distribution(0.0, 1.0);
    cauchy_distribution<double> cauchy_distribution(0.0, 1.0);
    poisson_distribution<int> poisson_distribution(10);
    uniform_real_distribution<double> uniform_real_distribution(-sqrt(3), sqrt(3));

    for (int n : sizes) { // Главный цикл, проходимся по каждому размеру n (10,100,1000)
        vector<double> normal_data, cauchy_data, laplace_data, poisson_data, uniform_data; // Создаем пустые массивы нужного размера для текущего n

        for (int i = 0; i < n; ++i) {
            normal_data.push_back(normal_distribution(gen));
            cauchy_data.push_back(cauchy_distribution(gen));
            laplace_data.push_back(generateLaplace(gen));
            // Пуассон генерирует целые числа, но мы сохраняем их как double для универсальной функции сохранения
            poisson_data.push_back(static_cast<double>(poisson_distribution(gen)));
            uniform_data.push_back(uniform_real_distribution(gen));
        }
    
        //В имени файла указываем тип распределения и размер массива
        saveToFile(normal_data, "normal_" + to_string(n) + ".csv");
        saveToFile(cauchy_data, "cauchy_" + to_string(n) + ".csv");
        saveToFile(laplace_data, "laplace_" + to_string(n) + ".csv");
        saveToFile(poisson_data, "poisson_" + to_string(n) + ".csv");
        saveToFile(uniform_data, "uniform_" + to_string(n) + ".csv");    
    }
    
    cout << "15 files were generated" << endl;
    return 0;
}