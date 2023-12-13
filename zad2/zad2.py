import pandas as pd
import functions as func
from importlib import reload
from scipy.stats import t
import numpy as np

reload(func)


def r_limit_func(n):
    t_student = t.ppf(1 - 0.05 / 2, n - 2)
    r_limit_value = np.sqrt((t_student ** 2) / (t_student ** 2 + n - 2))
    return r_limit_value


def check_y_xn_correlation(correlation_matrix, r_limit, data):
    for col in correlation_matrix.columns:
        if abs(correlation_matrix.loc[col, 'Pct_BF']) < r_limit:
            print("Odrzucono parametr: ", col, "[ r = ", round(abs(correlation_matrix.loc[col, 'Pct_BF']), 4), "]")
            data = data.drop(columns=col)
    return data


def check_xn_xn_correlation(data):
    flag = True

    while flag:
        flag = False
        # ponowne obliczenie macierzy korelacji
        correlation_matrix = data.corr()
        # współczynnik graniczny korelacji

        r_limit = r_limit_func(data.shape[1])

        # odrzucenie parametrów z wysoką korelacją x_n z x_n
        to_drop = None
        max_index_x = correlation_matrix.columns[1]
        max_index_y = correlation_matrix.columns[2]

        for col in correlation_matrix.columns[1:]:
            for col2 in correlation_matrix.columns[1:]:
                if col != col2:
                    if abs(correlation_matrix.loc[col, col2]) > abs(correlation_matrix.loc[max_index_x, max_index_y]):
                        max_index_x = col
                        max_index_y = col2

        if abs(correlation_matrix.loc[max_index_x, max_index_y]) > r_limit:
            flag = True
            if correlation_matrix.loc[max_index_x, 'Pct_BF'] > correlation_matrix.loc[max_index_y, 'Pct_BF']:
                print("Odrzucono parametr: ", max_index_y, "[ r = ",
                      round(abs(correlation_matrix.loc[max_index_y, max_index_x]), 4), ", r* =", round(r_limit, 4),
                      "]", )
                data = data.drop(columns=max_index_y)
            else:
                print("Odrzucono parametr: ", max_index_x, "[ r = ",
                      round(abs(correlation_matrix.loc[max_index_y, max_index_x]), 4), ", r* =", round(r_limit, 4),
                      "]", )
                data = data.drop(columns=max_index_x)

    return data


def prediction(x_pred, A, X_t, X, Se, body_fat_test):
    Yp = np.dot(x_pred, A)
    Yp = [round(i[0], 1) for i in Yp]
    results = list(zip(Yp, body_fat_test))
    print("\nPorównanie predykcji z danymi testowymi:")
    print("Predykcja, dana testowa")
    e_wzg_list = []

    for i, r in enumerate(results):
        e = round(r[1] - r[0], 1)
        e_wzg = round(abs(e) / r[1], 2) * 100
        e_wzg_list.append(e_wzg)
        print(r, " różnica: ", e, ", błąd wzg. różnicy: ", e_wzg, "%")
        Sp = np.dot(X_t, X)
        Sp = np.linalg.inv(Sp)
        Sp = np.dot(x_pred[i], Sp)
        Sp = np.dot(Sp, np.transpose(x_pred[i]))
        Sp = np.sqrt(1 + Sp)
        Sp = Se * Sp
        Sp = np.extract(1, Sp)[0]
        Vp = Sp / Yp[i] * 100
        print("      Średni błąd pred.: ", round(Sp, 2), ",  względny błąd pred: ", round(Vp, 2), "%")

    print("\nŚredni błąd procentowy różnicy między y_pred i y_test", np.mean(e_wzg_list))
    print("Max błąd procentowy różnicy między y_pred i y_test", np.max(e_wzg_list))


# załadowanie danych do data frame
data = pd.read_csv('dane.txt', delimiter='\t')
# y jako pierwsza kolumna
data = data[['Pct_BF', 'Density'] + list(data.columns[2:])]

# macierz korelacji
correlation_matrix = data.corr()
# współczynnik graniczny korelacji
r_limit_value = r_limit_func(data.shape[1])
print("Współczynnik graniczny korelacji:", r_limit_value)
# odrzucenie parametrów z niską korelacją y z x_n
print("\nOdrzucenie parametrów z niską korelacją y z x_n")
data = check_y_xn_correlation(correlation_matrix, r_limit_value, data)
# odrzucenie parametrów z wysoką korelacją x_n z x_n
print("\nOdrzucenie parametrów z wysoką korelacją x_n z x_n")
data = check_xn_xn_correlation(data)

print("\nPozostałe parametry:")
for i in data.columns:
    print(i)

# podzielenie danych
body_fat = data.Pct_BF
parametry = data.drop("Pct_BF", axis='columns')

# przekonwertowanie na listy
parametry = parametry.values.tolist()
body_fat = body_fat.tolist()

# podział na dane uczące i testowe
split_point = int(0.8 * len(parametry))
parametry, parametry_test = parametry[:split_point], parametry[split_point:]
body_fat, body_fat_test = body_fat[:split_point], body_fat[split_point:]

# Model liniowy
X, X_t, Y, A = func.build_matrices(parametry, body_fat, 1, True)
e, Se, Se2 = func.standard_deviation(X, A, Y, parametry, 1)
Sa = func.cov_matrix(Se2, X, X_t)
func.rates(e, Y, parametry[0], body_fat, Se)
x_pred = func.build_x_matrix(parametry_test, 1, True)
prediction(x_pred, A, X_t, X, Se, body_fat_test)

# 3) dodatkowo, na podstawie danych zaproponować model/-e dla innej/-ych
# zmiennej/-ych objaśnianej/-ących.

# Wybrana zmienna to Weight
print("\n\n############# Model liniowy dla zmiennej 'Weight'")

# załadowanie danych do data frame
data = pd.read_csv('dane.txt', delimiter='\t')
# y jako pierwsza kolumna
data = data[['Weight', 'Density', 'Pct_BF', 'Age'] + list(data.columns[4:])]

# macierz korelacji
correlation_matrix = data.corr()
# współczynnik graniczny korelacji
r_limit_value = r_limit_func(data.shape[1])
print("Współczynnik graniczny korelacji:", r_limit_value)
# odrzucenie parametrów z niską korelacją y z x_n
print("\nOdrzucenie parametrów z niską korelacją y z x_n")
data = check_y_xn_correlation(correlation_matrix, r_limit_value, data)
# odrzucenie parametrów z wysoką korelacją x_n z x_n
print("\nOdrzucenie parametrów z wysoką korelacją x_n z x_n")
data = check_xn_xn_correlation(data)

print("\nPozostałe parametry:")
for i in data.columns:
    print(i)
print("")

# podzielenie danych
weight = data.Weight
parametry = data.drop("Weight", axis='columns')

# przekonwertowanie na listy
parametry = parametry.values.tolist()
weight = weight.tolist()

# podział na dane uczące i testowe
split_point = int(0.8 * len(parametry))
parametry, parametry_test = parametry[:split_point], parametry[split_point:]
weight, weight_test = weight[:split_point], weight[split_point:]

# Model liniowy
X, X_t, Y, A = func.build_matrices(parametry, weight, 1, True)
e, Se, Se2 = func.standard_deviation(X, A, Y, parametry, 1)
Sa = func.cov_matrix(Se2, X, X_t)
func.rates(e, Y, parametry[0], weight, Se)
x_pred = func.build_x_matrix(parametry_test, 1, True)
prediction(x_pred, A, X_t, X, Se, weight_test)
