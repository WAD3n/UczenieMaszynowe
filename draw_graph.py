import matplotlib.pyplot as plt
import numpy as np


def draw_graph(array1,array2,axisx_name,axisy_name,graph_name,funtion1 = None,funtion2 = None):
    plt.title(graph_name)
    plt.xlabel(axisx_name)
    plt.ylabel(axisx_name)
    # dodanie wartosci nad punktami
    for i, (x, y) in enumerate(zip(array1, array2)):
        plt.text(x, y, f'{y}', fontsize=8, ha='center', va='bottom', color='black')
    # naniesienie puntkow na wykres
    plt.scatter(array1, array2)
    # utworzenie przestrzeni liniowej dla podstawienia do funkcji
    x = np.linspace(0, len(array1), 1000)
    # naniesienie funkcji na wykres
    plt.plot(x,funtion1(x))
    plt.plot(x,funtion2(x),color='red')
    plt.savefig(graph_name+'.jpg')
    plt.close()

