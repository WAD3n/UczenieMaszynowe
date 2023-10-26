import csv
import matplotlib.pyplot as plt
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

# Utworzenie wykresow
plt.title('rok - pracownicy')
for i, (x, y) in enumerate(zip(rok, pracownicy)):
    plt.text(x, y, f'{y}', fontsize=8, ha='center', va='bottom', color='black')
plt.scatter(rok, pracownicy)
plt.xlabel('rok')
plt.ylabel('pracownicy')
plt.savefig('rok-pracownicy.png')
plt.close()

plt.title('rok - przychod')
for i, (x, y) in enumerate(zip(rok, przychod)):
    plt.text(x, y, f'{y}', fontsize=8, ha='center', va='bottom', color='black')
plt.scatter(rok, przychod)
plt.xlabel('rok')
plt.ylabel('przychod')
plt.savefig('rok-przychod.png')
plt.close()

# Posortowanie danych według wartości zysku rosnąco
sorted_indices = sorted(range(len(zysk)), key=lambda _: zysk[_])
sorted_rok = [rok[i] for i in sorted_indices]
sorted_zysk = [zysk[i] for i in sorted_indices]

# Utworzenie wykresu
plt.title('rok - zysk')
for i, (x, y) in enumerate(zip(sorted_rok, sorted_zysk)):
    plt.text(x, y, f'{y}', fontsize=8, ha='center', va='bottom', color='black')
plt.scatter(sorted_rok, sorted_zysk)
plt.xlabel('rok')
plt.ylabel('zysk')
plt.savefig('rok-zysk.png')
plt.close()


def function(lista1, lista2):
    tmp = []
    for _ in lista1:
        tmp.append(pow(_, 2))
    macierz1 = []
    macierz2 = []
    macierz_niewaiadomych = []
    macierz1.append([sum(tmp), sum(lista1)])
    macierz1.append([sum(lista1), len(lista1)])
    print(macierz1)

    for _ in range(len(lista1)):
        tmp[_] = lista1[_]*lista2[_]
    macierz2.append(sum(tmp))
    macierz2.append(sum(lista2))
    print("\n", macierz2)

    macierz_niewaiadomych.append(
        macierz1[0][0]*macierz2[0]+macierz1[0][0]*macierz2[1]+macierz1[0][1]*macierz2[0]+macierz1[0][1]*macierz2[1])
    macierz_niewaiadomych.append(
        macierz1[1][0] * macierz2[0] + macierz1[1][0] * macierz2[1] + macierz1[1][1] * macierz2[0] + macierz1[1][1] *
        macierz2[1])
    print("\n",macierz_niewaiadomych)


function(rok, pracownicy)
