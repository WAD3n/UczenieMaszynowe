import numpy as np
from pandas import read_excel
from scipy.stats import t


def load_data_from_txt(name, okres):
    period = []
    users = []

    with open(name) as f:
        lines = f.readlines()

    for l in lines:
        splited_line = l.split()
        quarter = int(splited_line[0][1:2])
        year = int(splited_line[1][1:3])
        if okres == "kwartał":
            period.append((year - 8) * 4 + quarter % 5)
            users.append(int(splited_line[2]))
        elif okres == "rok":
            if quarter == 4:
                period.append(year - 8)
                users.append(int(splited_line[2]))

    return period, users


def load_data_from_excel():
    dane = read_excel(r"dane2.xlsx")
    return dane['Rok'].tolist(), dane['Przychód w mln $'].tolist(), dane['Zysk w mln $'].tolist(), dane[
        'Zatrudnienie'].tolist()


def build_x_matrix(x, degree):
    def build_x(i):
        line = []
        for d in range(degree):
            line.append(i ** (d + 1))
        line.append(1)
        return line

    X = list(map(build_x, x))
    X = np.array(X)
    return X


def build_matrices(x, y, degree):
    X = build_x_matrix(x, degree)
    X_t = X.transpose()

    Y = list(map(lambda n: [n], y))
    Y = np.array(Y)

    A = np.dot(X_t, X)
    A = np.linalg.inv(A)
    A = np.dot(A, X_t)
    A = np.dot(A, Y)

    return X, X_t, Y, A


def quarter_to_index(quarter, year):
    return (year - 2008) * 4 + quarter % 5


def year_to_index(year):
    return year - 2006


def y_model(A, x):
    return A[0] * x + A[1]


def y_model(A, x, degree):
    if degree == 0:
        return [A[-1] for _ in x]
    else:
        return [A[degree - 1] * (xi ** degree) + y_model(A, x, degree - 1)[i] for i, xi in enumerate(x)]


def przychod_model(b, a, x):
    return [b * (a ** xi) for xi in x]


def standard_deviation(X, A, Y, x, k=1):
    # Odchylenie standardowe składnika resztowego
    e = np.array(Y - np.dot(X, A))
    # print(e)
    # k- liczba kolumnw macierzy X / liczba X we wzorze
    Se2 = np.dot(e.transpose(), e) / (len(x) - (k + 1))
    Se = np.sqrt(Se2)
    Se = np.extract(1, Se)[0]
    print("Odchylenie standardowe składnika resztowego:", round(Se, 2))
    return e, Se, Se2


def cov_matrix(Se2, X, X_t):
    cov_A = Se2 * (np.linalg.inv(np.dot(X_t, X)))
    # print("Standardowy błąd szasunku parametru a1: ", round(np.sqrt(cov_A[0][0]), 2))
    Sa2 = np.diag(cov_A)
    Sa = list(map(lambda x: np.sqrt(x), Sa2))
    return Sa


def rates(e, Y, x, y, Se):
    # Współczynnik zbieżności
    wsp_zb_2 = np.dot(e.transpose(), e) / (np.dot(Y.transpose(), Y) - (len(x) * (np.average(y)) ** 2))
    wsp_zb_2 = np.extract(1, wsp_zb_2)[0]
    wsp_zb_2 = round(wsp_zb_2, 4)
    print("Współczynnik zbieżności: ", wsp_zb_2)

    # Współczynnik determinacji
    print("Współczynnik determinacji: ", 1 - wsp_zb_2)

    # Współczynnik zmienności losowej
    We = Se / np.average(y) * 100
    print("Współczynnik zmienności losowej: ", round(We, 2), "%")


def prediction(x_pred, A, X_t, X, Se):
    Yp = np.dot(x_pred, A)
    Yp = np.extract(1, Yp)[0]
    print("Predykcja dla x = ", x_pred[0][0], " wynosi: ", round(Yp, 2))
    Sp = np.dot(X_t, X)
    Sp = np.linalg.inv(Sp)
    Sp = np.dot(x_pred, Sp)
    Sp = np.dot(Sp, np.transpose(x_pred))
    Sp = np.sqrt(1 + Sp)
    Sp = Se * Sp
    Sp = np.extract(1, Sp)[0]
    print("  Średni błąd predykacji wynosi: ", round(Sp, 2))
    Vp = Sp / Yp * 100
    print("  Względny błąd predykcji wynosi: ", round(Vp, 2), "%")
    return Yp


def badanie_istotnosci(A, Sa, n, k):
    once_again = 1

    while once_again:
        I_i = []
        critical_value = t.ppf(1 - 0.05 / 2, n - k - 1)

        for i in range(len(A)):
            I_i.append(abs(A[i][0]) / Sa[i])

        min_I_i = min(x for x in I_i if x != 0)

        if min_I_i < critical_value:
            index = I_i.index(min_I_i)
            A[index] = 0
            print("Element a[", index, "] uznano za nieistotny")
            k = k - 1
        else:
            once_again = 0
