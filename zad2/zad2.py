import pandas as pd
import functions as func
from importlib import reload
import numpy as np

reload(func)

# załadowanie danych do data frame
data = pd.read_csv('dane.txt', delimiter='\t')

# podzielenie danych
body_fat = data.Pct_BF
parametry = data.drop("Pct_BF", axis='columns')

# przekonwertowanie na listy
parametry = parametry.values.tolist()
body_fat = body_fat.tolist()

X, X_t, Y, A = func.build_matrices(parametry, body_fat, 1, True)
e, Se, Se2 = func.standard_deviation(X, A, Y, parametry, 1)
Sa = func.cov_matrix(Se2, X, X_t)
a = np.linalg.inv(np.dot(X_t, X))
print(a)

np.savetxt('doexcela.txt', X, delimiter='\t', header='', comments='', fmt='%.4f')

for i in range(len(a)):
    print(round(a[i][i], 4))

# d = np.diag(a)
# print(d)

# x_pred = func.build_x_matrix(
#     [1.0708, 23.0, 154.25, 67.75, 36.2, 93.1, 85.2, 33.543307, 94.5, 59.0, 37.3, 21.9, 32.0, 27.4, 17.1, ], 1)
# func.prediction(x_pred, A, X_t, X, Se)

# Proszę:
# 1) zbudować model pozwalający przewidzieć %bodyfat na podstawie innych
# zmiennych. Procent tłuszczu ciała każdego (%bodyfat, PctBF) znajduje się w
# drugiej kolumnie danych.
# 2) przed budową modelu proszę zaproponować/wybrać procedurę eliminacji
# zmiennych wraz z uzasadnieniem.

# 3) dodatkowo, na podstawie danych zaproponować model/-e dla innej/-ych
# zmiennej/-ych objaśnianej/-ących.
# 4) przygotować w formie pisemnej wyczerpujące sprawozdanie z wykonania zadania
# (!), które wraz z innymi plikami (wymaganymi do weryfikacji rozwiązania zadania)
# należy umieścić w prywatnym notesie zespołu MS Teams w sekcji „Laboratorium”
# w odpowiednio nazwanej stronie, np. Sprawozdanie z lab. 1
