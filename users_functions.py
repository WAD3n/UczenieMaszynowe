import numpy as np


def load_data_from_txt(name):
    quarter_index = []
    users = []

    with open(name) as f:
        lines = f.readlines()

    for l in lines:
        splited_line = l.split()
        quarter = int(splited_line[0][1:2])
        year = int(splited_line[1][1:3])
        quarter_index.append((year - 8) * 4 + quarter % 5)
        users.append(int(splited_line[2]))

    return quarter_index, users


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
