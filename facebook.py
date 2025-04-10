from importlib import reload
import numpy as np
# dodanie modulow czastkowych
import draw_graph as draw
import modele_funkcji as mf
import functions as func

reload(func)
reload(draw)
reload(mf)

########################################## Model liniowy   - liczba użytkowników od kwartału
quarter_index, users = func.load_data_from_txt('dane', "kwartał")
X, X_t, Y, A = func.build_matrices(quarter_index, users, 1)
e, Se, Se2 = func.standard_deviation(X, A, Y, quarter_index)
func.cov_matrix(Se2, X, X_t)
func.rates(e, Y, quarter_index, users, Se)
x_pred = func.build_x_matrix([func.quarter_to_index(1, 2018)], 1)
func.prediction(x_pred, A, X_t, X, Se)
x_pred = func.build_x_matrix([func.quarter_to_index(2, 2020)], 1)
func.prediction(x_pred, A, X_t, X, Se)

draw.make_plot(quarter_index, users, 'Liczba użytkowniów facebooka', 'kwartał', 'liczba użytkowników',
               func.y_model(A, quarter_index, 1), 'uzytkownicy_kwartal.jpg')

########################################### Wielomianowy / wykładniczy - liczba pracowników od roku
rok, przychod, zysk, pracownicy = func.load_data_from_excel()
new_rok = [func.year_to_index(n) for n in rok]

draw.draw_graph(new_rok, pracownicy, 'rok', 'liczba pracownikow', 'pracownicy',
                mf.wyznacz_funkcje_wielomianowa(new_rok, pracownicy, 3),
                mf.wyznacz_funkcje_wykladnicza(new_rok, pracownicy))

########################################### Wielomianowy / wykładniczy - przychód od roku
draw.draw_graph(new_rok, przychod, 'rok', 'przychod', 'przychod', mf.wyznacz_funkcje_wielomianowa(new_rok, przychod, 3),
                mf.wyznacz_funkcje_wykladnicza(new_rok, przychod))

########################################### Wielomianowy / wykładniczy - zysk od roku
draw.draw_graph(new_rok, zysk, 'rok', 'zysk', 'zysk', mf.wyznacz_funkcje_wielomianowa(new_rok, zysk, 3),
                mf.wyznacz_funkcje_wykladnicza(new_rok, zysk))

########################################### Wykładniczy - przychód od roku
# model y = b*a^x
# linearyzacja  log_y = log_b + x * log_A
# zał.: przychod_l = log_y , a_0 = log_b , a_1 = log_A
# zatem: przychod_l = x * a_1 + a_0

print("\n")
przychod_l = [np.log(x) for x in przychod]
X, X_t, Y, A = func.build_matrices(new_rok, przychod_l, 1)
e, Se, Se2 = func.standard_deviation(X, A, Y, new_rok)
func.cov_matrix(Se2, X, X_t)
func.rates(e, Y, new_rok, przychod_l, Se)
a = np.e ** A[0][0]
b = np.e ** A[1][0]

print(func.przychod_model(b, a, [9]))

draw.make_plot(new_rok, przychod, 'Przychod', 'rok', 'przychód',
               func.przychod_model(b, a, new_rok), 'przychod2.jpg')

########################################### Wielomianowy - zatrudnienie od roku
print("\n")

print("Wielomianowy - zatrudnienie od roku")
X, X_t, Y, A = func.build_matrices(new_rok, pracownicy, 3)
e, Se, Se2 = func.standard_deviation(X, A, Y, new_rok, 3)
Sa = func.cov_matrix(Se2, X, X_t)
func.rates(e, Y, new_rok, pracownicy, Se)
func.badanie_istotnosci(A, Sa, len(X), 3)
e, Se, Se2 = func.standard_deviation(X, A, Y, new_rok)
Sa = func.cov_matrix(Se2, X, X_t)

x_pred = func.build_x_matrix([func.year_to_index(2017)], 3)
func.prediction(x_pred, A, X_t, X, Se)

draw.make_plot(new_rok, pracownicy, 'Zatrudnienie', 'rok', 'liczba pracowników',
               func.y_model(A, new_rok, 3), 'zatrudnienie.jpg')

########################################### Liczba użytkowników od roku

print("")
year, users = func.load_data_from_txt('dane', "rok")
print(year)
X, X_t, Y, A = func.build_matrices(year, users, 1)
e, Se, Se2 = func.standard_deviation(X, A, Y, year)
func.cov_matrix(Se2, X, X_t)
func.rates(e, Y, year, users, Se)
rok_pred = func.build_x_matrix([10], 1)
uzytk_pred = func.prediction(rok_pred, A, X_t, X, Se)
print(uzytk_pred)

draw.make_plot(year, users, 'Liczba użytkowniów facebooka', 'rok', 'liczba użytkowników',
               func.y_model(A, year, 1), 'uzytkownicy_rok.jpg')

########################################### Przychód od liczby użytkowników

przychod.pop(0)
przychod.pop(0)
X, X_t, Y, A = func.build_matrices(users, przychod, 1)

draw.draw_graph(users, przychod, 'uzytkownicy', 'przychod', 'przychod_od_uzytkownikow',
                mf.wyznacz_funkcje_wielomianowa(users, przychod, 3),
                mf.wyznacz_funkcje_wykladnicza(users, przychod))

### WYKRES NA PODSTAWIE PRZEWIDZIANEJ WARTOSCI PODSTAWIONY DO FUNKCJI WIELOMIANOWEJ
# helper = mf.wyznacz_funkcje_wielomianowa(users, przychod, 3),
# users.append(uzytk_pred)
# k = helper[0]
# przychod.append(k(uzytk_pred).__floor__())
#
# draw.draw_graph(users, przychod, 'uzytkownicy', 'przychod', 'przychod_od_uzytkownikow_pred_wielo',
#                 mf.wyznacz_funkcje_wielomianowa(users, przychod, 3),
#                 mf.wyznacz_funkcje_wykladnicza(users, przychod))


### WYKRES NA PODSTAWIE PRZEWIDZANEJ WARTOSCI PODSTAWIONY DO FUNCKJI WYKLADNICZEJ
helper = mf.wyznacz_funkcje_wykladnicza(users, przychod),
users.append(uzytk_pred)
k = helper[0]
przychod.append(k(uzytk_pred).__floor__())

draw.draw_graph(users, przychod, 'uzytkownicy', 'przychod', 'przychod_od_uzytkownikow_pred_wykl',
                mf.wyznacz_funkcje_wielomianowa(users, przychod, 3),
                mf.wyznacz_funkcje_wykladnicza(users, przychod))