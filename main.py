import csv
import numpy as np
import matplotlib.pyplot as plt
import draw_graph


# Utworzenie zmiennej csvreader która zawiera kontekst pliku csv a nastepnie
# utworzenie zmiennej list ktora jest lista wierszy wczesniej otworzonego pliku
file = open('zarobki.csv')
csvreader = csv.reader(file)
lista = []
for line in csvreader:
    lista.append(line)
# Wyrzucenie wiersza zawierajaca nazwy kolumn
lista.pop(0)

# Stworzenie 4 list odpowiadajacych kolumnom pliku csv i umieszczenie w nich danych
rok = []
pracownicy = []
zysk = []
przychod = []
for element in lista:
    rok.append(int(element[0]))
    przychod.append(int(element[1]))
    zysk.append(int(element[2]))
    pracownicy.append(int(element[3]))

stopien = 4

new_rok = []
for _ in range(len(rok)):
    new_rok.append(_)

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
    ln_y = np.log(np.abs(y))
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

# Utworzenie wykresow
funkcja, wspolczynniki = wyznacz_funkcje_wielomianowa(new_rok, pracownicy, stopien)
funkcja_wykladnicza,a,b = wyznacz_funkcje_wykladnicza(new_rok,pracownicy)
# plt.title('rok - pracownicy')
# for i, (x, y) in enumerate(zip(new_rok, pracownicy)):
#     plt.text(x, y, f'{y}', fontsize=8, ha='center', va='bottom', color='black')
# plt.scatter(new_rok, pracownicy)
# plt.xlabel('rok')
# plt.ylabel('pracownicy')
# x = np.linspace(0,len(rok), 1000)
# y = funkcja(x)
# plt.plot(x, y)
# plt.plot(x,funkcja_wykladnicza(x),color='red')
# plt.savefig('rok-pracownicy.png')
# plt.close()
draw_graph.draw_graph(new_rok,pracownicy,'rok','pracownicy','pracownicy',funkcja,funkcja_wykladnicza)

plt.title('rok - przychod')
funkcja_przychod, wspolczynniki_przychod = wyznacz_funkcje_wielomianowa(new_rok, przychod, stopien)
funkcja_wykladnicza_przychod , c, d = wyznacz_funkcje_wykladnicza(new_rok, przychod)
for i, (x, y) in enumerate(zip(new_rok, przychod)):
    plt.text(x, y, f'{y}', fontsize=8, ha='center', va='bottom', color='black')
plt.scatter(new_rok, przychod)
x_przychod = np.linspace(0, len(new_rok), 1000)
plt.xlabel('rok')
plt.ylabel('przychod')
plt.plot(x_przychod, funkcja_przychod(x_przychod))
plt.plot(x_przychod, funkcja_wykladnicza_przychod(x_przychod),color='red')
plt.savefig('rok-przychod.png')
plt.close()

# Posortowanie danych według wartości zysku rosnąco
sorted_indices = sorted(range(len(zysk)), key=lambda _: zysk[_])
sorted_rok = [rok[i] for i in sorted_indices]
sorted_zysk = [zysk[i] for i in sorted_indices]

# Utworzenie wykresu
plt.title('rok - zysk')
funkcja_zysk, wspolczynniki_zysk = wyznacz_funkcje_wielomianowa(new_rok, zysk, stopien)
funkcja_wykladnicza_zysk , k, l = wyznacz_funkcje_wykladnicza(new_rok, zysk)
for i, (x, y) in enumerate(zip(new_rok, sorted_zysk)):
    plt.text(x, y, f'{y}', fontsize=8, ha='center', va='bottom', color='black')
plt.scatter(new_rok, sorted_zysk)
plt.xlabel('rok')
plt.ylabel('zysk')
x_zysk = np.linspace(0,len(new_rok), 1000)
plt.plot(x_zysk, funkcja_przychod(x_zysk))
plt.plot(x_zysk, funkcja_wykladnicza_zysk(x_zysk),color='red')
plt.savefig('rok-zysk.png')
plt.close()

