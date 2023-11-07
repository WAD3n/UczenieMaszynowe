import numpy as np

def wyznacz_funkcje_wielomianowa(x, y, stopien):
    if len(x) != len(y):
        raise ValueError("Liczba punktów musi być taka sama.")
    A = np.vander(x, N=stopien + 1, increasing=True)
    b = y
    wspolczynniki = np.linalg.lstsq(A, b, rcond=None)[0]
    def funkcja_wielomianowa(x):
        return sum(wspolczynniki[i] * x ** i for i in range(stopien + 1))

    return funkcja_wielomianowa#, wspolczynniki

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

    return funkcja_wykladnicza#, a, b