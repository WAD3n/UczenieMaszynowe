import matplotlib.pyplot as plt
import numpy as np
from pandas import read_excel

fig, ax = plt.subplots(2, 2, figsize=(30, 16))


def load_data():
    quarter_index = []
    users = []

    with open('dane') as f:
        lines = f.readlines()

    for l in lines:
        splited_line = l.split()
        quarter = int(splited_line[0][1:2])
        year = int(splited_line[1][1:3])
        quarter_index.append((year - 8) * 4 + quarter % 5)
        users.append(int(splited_line[2]))

    return quarter_index, users


def make_plot(plot_index1, plot_index2, x, y, title, xlabel, ylabel, model):
    ax[plot_index1, plot_index2].scatter(x, y)
    ax[plot_index1, plot_index2].set_title(title)
    ax[plot_index1, plot_index2].set_xlabel(xlabel)
    ax[plot_index1, plot_index2].set_ylabel(ylabel)
    ax[plot_index1, plot_index2].plot(x, model, 'g-', linewidth=2.0)


def build_matrices(x, y):
    X = list(map(lambda n: [n, 1], x))
    X = np.array(X)
    X_t = X.transpose()
    # print(X)

    Y = list(map(lambda n: [n], y))
    Y = np.array(Y)
    # print(Y)

    A = np.dot(X_t, X)
    A = np.linalg.inv(A)
    # print(A)
    A = np.dot(A, X_t)
    A = np.dot(A, Y)

    return X, X_t, Y, A


def quarter_to_index(quarter, year):
    return (year - 2008) * 4 + quarter % 5


def y_model(index):
    return A[0] * index + A[1]


def standard_deviation(X, A, x, k=1):
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
    # Macierz kowariancji
    cov_A = Se2 * (np.linalg.inv(np.dot(X_t, X)))
    # print(cov_A)
    print("Standardowy błąd szasunku parametru a1: ", round(np.sqrt(cov_A[0][0]), 2))
    print("Standardowy błąd szasunku parametru a0: ", round(np.sqrt(cov_A[1][1]), 2))
    return cov_A


def rates(e, Y, x, y):
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


def prediction(quarter, year, X_t, X, Se, ):
    index = quarter_to_index(quarter, year)
    Yp = np.extract(1, y_model(index))[0]
    print("Predykcja dla kwartału ", index, " wynosi: ", round(Yp, 2))
    Sp = np.dot(X_t, X)
    Sp = np.linalg.inv(Sp)
    x_pred = np.array([index, 1])
    Sp = np.dot(x_pred, Sp)
    Sp = np.dot(Sp, np.transpose(x_pred))
    Sp = np.sqrt(1 + Sp)
    Sp = Se * Sp
    Sp = np.extract(1, Sp)[0]
    print("  Średni błąd predykacji wynosi: ", round(Sp, 2))
    Vp = Sp / Yp * 100
    print("  Względny błąd predykcji wynosi: ", round(Vp, 2), "%")


quarter_index, users = load_data()
X, X_t, Y, A = build_matrices(quarter_index, users)
e, Se, Se2 = standard_deviation(X, A, quarter_index)
cov_matrix(Se2, X, X_t)
rates(e, Y, quarter_index, users)
prediction(1, 2018, X_t, X, Se)
prediction(4, 2018, X_t, X, Se)
make_plot(0, 0, quarter_index, users, 'Liczba użytkowniów facebooka', 'kwartał', 'liczba użytkowników',
          y_model(quarter_index))


########################################### Nieliniowe #################
def load_data2():
    dane = read_excel(r"dane2.xlsx")
    return dane['Rok'].tolist(), dane['Przychód w mln $'].tolist(), dane['Zysk w mln $'].tolist(), dane[
        'Zatrudnienie'].tolist()


def wyznacz_funkcje_wielomianowa(x, y, stopien):
    if len(x) != len(y):
        raise ValueError("Liczba punktów musi być taka sama.")
    A = np.vander(x, N=stopien + 1, increasing=True)
    b = y
    wspolczynniki = np.linalg.lstsq(A, b, rcond=None)[0]

    def funkcja_wielomianowa(x):
        return sum(wspolczynniki[i] * x ** i for i in range(stopien + 1))

    return funkcja_wielomianowa, wspolczynniki


def wyznacz_funkcje_wykladnicza(x, y):
    if len(x) != len(y):
        raise ValueError("Liczba punktów musi być taka sama.")
    ln_y = np.log(y)
    # Stwórz macierz A i wektor b
    A = np.vstack([np.ones(len(x)), x]).T
    b = ln_y
    # Rozwiąż układ równań liniowych A * x = b
    ln_a, b_value = np.linalg.lstsq(A, b, rcond=None)[0]
    a = np.exp(ln_a)
    b = b_value

    def funkcja_wykladnicza(x):
        return a * np.exp(b * x)

    return funkcja_wykladnicza, a, b


stopien = 6
rok, przychod, zysk, pracownicy = load_data2()
new_rok = [n - 2006 for n in rok]

# Utworzenie wykresow
funkcja, wspolczynniki = wyznacz_funkcje_wielomianowa(new_rok, pracownicy, stopien)
funkcja_wykladnicza, a, b = wyznacz_funkcje_wykladnicza(new_rok, pracownicy)
ax[0, 1].set_title('rok - pracownicy')
for i, (x, y) in enumerate(zip(new_rok, pracownicy)):
    ax[0, 1].text(x, y, f'{y}', fontsize=8, ha='center', va='bottom', color='black')
ax[0, 1].scatter(new_rok, pracownicy)
ax[0, 1].set_xlabel('rok')
ax[0, 1].set_ylabel('pracownicy')
x = np.linspace(0, len(rok), 1000)
y = funkcja(x)
ax[0, 1].plot(x, y)
ax[0, 1].plot(x, funkcja_wykladnicza(x), color='red')

X, X_t, Y, A = build_matrices(new_rok, pracownicy)
e, Se, Se2 = standard_deviation(X, A, new_rok)
Sa = cov_matrix(Se2, X, X_t)

print(A)
print(Sa)


def badanie_istotnosci(A, Sa):
    for n, a in enumerate(A):
        I = a / Sa[n][n]
        print(I)


badanie_istotnosci(wspolczynniki, Sa)

ax[1, 0].set_title('rok - przychod')
funkcja_przychod, wspolczynniki_przychod = wyznacz_funkcje_wielomianowa(rok, przychod, stopien)
for i, (x, y) in enumerate(zip(rok, przychod)):
    ax[1, 0].text(x, y, f'{y}', fontsize=8, ha='center', va='bottom', color='black')
ax[1, 0].scatter(rok, przychod)
x_przychod = np.linspace(2007, 2017, 1000)
ax[1, 0].set_xlabel('rok')
ax[1, 0].set_ylabel('przychod')
ax[1, 0].plot(x_przychod, funkcja_przychod(x_przychod))

# Posortowanie danych według wartości zysku rosnąco
sorted_indices = sorted(range(len(zysk)), key=lambda _: zysk[_])
sorted_rok = [rok[i] for i in sorted_indices]
sorted_zysk = [zysk[i] for i in sorted_indices]

# Utworzenie wykresu
ax[1, 1].set_title('rok - zysk')
funkcja_zysk, wspolczynniki_zysk = wyznacz_funkcje_wielomianowa(rok, zysk, stopien)
for i, (x, y) in enumerate(zip(sorted_rok, sorted_zysk)):
    ax[1, 1].text(x, y, f'{y}', fontsize=8, ha='center', va='bottom', color='black')
ax[1, 1].scatter(sorted_rok, sorted_zysk)
ax[1, 1].set_xlabel('rok')
ax[1, 1].set_ylabel('zysk')
x_zysk = np.linspace(2007, 2017, 1000)
ax[1, 1].plot(x_zysk, funkcja_zysk(x_zysk))

# model y = b*a^x
# linearyzacja  log_y = log_b + x * log_A
# zał.: przychod_l = log_y , a_0 = log_b , a_1 = log_A
# zatem: przychod_l = x * a_1 + a_0

# przychod_l = [np.log(x) for x in przychod]
# X, X_t, Y, A = build_matrices(rok, przychod_l)
# e, Se, Se2 = standard_deviation(X, A, rok)
# cov_matrix(Se2, X, X_t)
# rates(e, Y, rok, przychod_l)
# a1 = A[0][0]
# a0 = A[1][0]
# a = np.e ** a1
# b = np.e ** a0
# print(a)
# print(b)
#
# pred = przychod_model(b, a, 11)
# print(pred)

# X, X_t, Y, A = build_matrices(index, users)
# e, Se, Se2 = standard_deviation()
# cov_matrix()
# rates()
# prediction(1, 2018, X_t, X, Se)
# prediction(4, 2018, X_t, X, Se)


# make_plot(1, rok, przychod, 'Przychód', 'rok', 'przychód [$]', przychod)
######################################################################

plt.show()
