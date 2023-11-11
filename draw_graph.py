import matplotlib.pyplot as plt
import numpy as np


def make_plot(x, y, title, xlabel, ylabel, model):
    plt.scatter(x, y)
    # plt.subtitle(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(x, model, 'g-', linewidth=2.0)
    plt.savefig('uzytkownicy.jpg')
    plt.close()


def draw_graph(array1, array2, axisx_name, axisy_name, graph_name, funtion1=None, funtion2=None):
    sorted_indices = sorted(range(len(array2)), key=lambda _: array2[_])
    sorted_array1 = [array1[i] for i in sorted_indices]
    sorted_array2 = [array2[i] for i in sorted_indices]
    plt.title(graph_name)
    plt.xlabel(axisx_name)
    plt.ylabel(axisy_name)
    # dodanie wartosci nad punktami
    for i, (x, y) in enumerate(zip(array1, array2)):
        plt.text(x, y, f'{y}', fontsize=8, ha='center', va='bottom', color='black')
    # naniesienie puntkow na wykres
    plt.scatter(array1, array2)
    # utworzenie przestrzeni liniowej dla podstawienia do funkcji
    x = np.linspace(0, len(array1), 1000)
    # naniesienie funkcji na wykres
    if funtion1 != None:
        plt.plot(x, funtion1(x))
    if funtion2 != None:
        plt.plot(x, funtion2(x), color='red')
    plt.savefig(graph_name + '.jpg')
    plt.close()
