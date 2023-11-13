from importlib import reload
import numpy as np
# dodanie modulow czastkowych
import draw_graph as draw
import modele_funkcji as mf
import functions as func

reload(func)
reload(draw)

########################################## Model liniowy   - liczba użytkowników od kwartału
quarter_index, users = func.load_data_from_txt('dane')
X, X_t, Y, A = func.build_matrices(quarter_index, users, 1)
e, Se, Se2 = func.standard_deviation(X, A, Y, quarter_index)
func.cov_matrix(Se2, X, X_t)
func.rates(e, Y, quarter_index, users, Se)
x_pred = func.build_x_matrix([func.quarter_to_index(1, 2018)], 1)
func.prediction(x_pred, A, X_t, X, Se)

draw.make_plot(quarter_index, users, 'Liczba użytkowniów facebooka', 'kwartał', 'liczba użytkowników',
               func.y_model(A, quarter_index, 1), 'uzytkownicy.jpg')

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

X, X_t, Y, A = func.build_matrices(new_rok, pracownicy, 3)
e, Se, Se2 = func.standard_deviation(X, A, Y, new_rok)
Sa = func.cov_matrix(Se2, X, X_t)
func.rates(e, Y, new_rok, pracownicy, Se)
func.badanie_istotnosci(A, Sa, len(X), 3)
e, Se, Se2 = func.standard_deviation(X, A, Y, new_rok)
Sa = func.cov_matrix(Se2, X, X_t)

x_pred = func.build_x_matrix([func.year_to_index(2017)], 3)
func.prediction(x_pred, A, X_t, X, Se)

draw.make_plot(new_rok, pracownicy, 'Zatrudnienie', 'rok', 'liczba pracowników',
               func.y_model(A, new_rok, 3), 'zatrudnienie.jpg')

# można jeszcze podzielić dane na treningowe i testowe
