import pandas as pd
import functions as func
from importlib import reload
from scipy.stats import t
import numpy as np
import seaborn as sns
from statistics import median
import matplotlib.pyplot as plt

reload(func)


def r_limit_func(n):
    t_student = t.ppf(1 - 0.05 / 2, n - 2)
    r_limit_value = np.sqrt((t_student ** 2) / (t_student ** 2 + n - 2))
    return r_limit_value


def check_y_xn_correlation(correlation_matrix, r_limit, data, y):
    for col in correlation_matrix.columns:
        if abs(correlation_matrix.loc[col, 'Pct_BF']) < r_limit:
            print("Odrzucono parametr: ", col, "[ r = ", round(abs(correlation_matrix.loc[col, y]), 4), "]")
            data = data.drop(columns=col)
    return data


def check_xn_xn_correlation(correlation_matrix, r_limit, data, y):
    # dla każdego x_n z x_n z macierzy korelacji zostanie sprawdzone, czy podane parametry istnieją w zbiorze data
    # czy nie jest to korelacja tej samej zmiennej oraz czy korelacja jest większa od współczynnika graniczengo korelacji

    for col in correlation_matrix.columns[1:]:
        for col2 in correlation_matrix.columns[1:]:
            if col in data.columns and col2 in data.columns:
                if col != col2 and abs(correlation_matrix.loc[col, col2]) > r_limit:
                    if abs(correlation_matrix.loc[col, y]) > abs(correlation_matrix.loc[col2, y]):
                        print("Odrzucono parametr: ", col2, "[ r = ",
                              round(abs(correlation_matrix.loc[col, col2]), 4), "korelacja z ", col, "]")
                        data = data.drop(columns=col2)
                    else:
                        print("Odrzucono parametr: ", col, "[ r = ",
                              round(abs(correlation_matrix.loc[col, col2]), 4), "korelacja z ", col2, "]")
                        data = data.drop(columns=col)

    return data


def prediction(x_pred, A, y_test):
    Yp = np.dot(x_pred, A)
    Yp = [round(i[0], 1) for i in Yp]
    results = list(zip(Yp, y_test))
    print("\nPorównanie predykcji z danymi testowymi:")
    print("Predykcja, dana testowa")
    e_wzg_list = []

    for i, r in enumerate(results):
        e = round(r[1] - r[0], 1)
        e_wzg = round(abs(e) / r[1] * 100, 2)
        e_wzg_list.append(e_wzg)
        print(r, " różnica: ", e, ", błąd procentowy różnicy: ", e_wzg, "%")

    print("\nŚredni błąd procentowy różnicy między y_pred i y_test", round(np.mean(e_wzg_list), 1))
    print("Max błąd procentowy różnicy między y_pred i y_test", round(np.max(e_wzg_list), 2), "%")
    print("Mediana różnicy między y_pred i y_test", round(median(e_wzg_list), 2), "%")


def build_model(data, y_name):
    # podział na dane uczące i testowe losowo w proporcji 7:3
    data_test = data.sample(frac=0.3, random_state=42)
    data = data.drop(data_test.index)

    # macierz korelacji
    correlation_matrix = data.corr()
    sns.set(style="white")
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5)
    plt.title("Macierz korelacji")
    plt.show()

    # współczynnik graniczny korelacji
    r_limit_value = r_limit_func(data.shape[0])
    print("Współczynnik graniczny korelacji:", r_limit_value)
    # odrzucenie parametrów z niską korelacją y z x_n
    print("\nOdrzucenie parametrów z niską korelacją y z x_n")
    data = check_y_xn_correlation(correlation_matrix, r_limit_value, data, y_name)
    # ponowne obliczenie macierzy korelacji
    correlation_matrix = data.corr()
    # odrzucenie parametrów z wysoką korelacją x_n z x_n
    print("\nOdrzucenie parametrów z wysoką korelacją x_n z x_n")
    data = check_xn_xn_correlation(correlation_matrix, r_limit_value, data, y_name)

    print("\nPozostałe parametry:")
    for i in data.columns:
        print(i)
    print("")

    # podzielenie danych uczących
    y_learning = data[y_name]
    x_learning = data.drop(y_name, axis='columns')
    # przekonwertowanie na listy
    x_learning = x_learning.values.tolist()
    y_learning = y_learning.tolist()

    # podzielenie danych uczących
    y_test = data_test[y_name]
    x_test = data_test[data.columns].drop(y_name, axis='columns')
    # przekonwertowanie na listy
    x_test = x_test.values.tolist()
    y_test = y_test.tolist()

    # Model liniowy
    X, X_t, Y, A = func.build_matrices(x_learning, y_learning, 1, True)
    e, Se, Se2 = func.standard_deviation(X, A, Y, x_learning, 1)
    func.cov_matrix(Se2, X, X_t)
    func.rates(e, Y, x_learning[0], y_learning, Se)
    x_pred = func.build_x_matrix(x_test, 1, True)
    prediction(x_pred, A, y_test)


# załadowanie danych do data frame
data = pd.read_csv('dane.txt', delimiter='\t')
# y jako pierwsza kolumna
data = data[['Pct_BF', 'Density'] + list(data.columns[2:])]

build_model(data, 'Pct_BF')

# 3) dodatkowo, na podstawie danych zaproponować model/-e dla innej/-ych
# zmiennej/-ych objaśnianej/-ących.

# Wybrana zmienna to Weight
print("\n\n############# Model liniowy dla zmiennej 'Weight'")

# załadowanie danych do data frame
data = pd.read_csv('dane.txt', delimiter='\t')
# y jako pierwsza kolumna
data = data[['Weight', 'Density', 'Pct_BF', 'Age'] + list(data.columns[4:])]

build_model(data, 'Weight')
