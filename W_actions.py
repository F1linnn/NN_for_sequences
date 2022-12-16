import random
import numpy as np


def generate_W1(hide_amount: int, size_in: int):
    W = []
    for h in range(hide_amount):
        lst = []
        for weight in range(size_in):
            lst.append(random.uniform(-1, 1))
        W.append(lst)
    return np.array(W)


def generate_W2(hide_amount: int, amount_out: int):
    W = []
    for h in range(hide_amount):
        lst = []
        for weight in range(amount_out):
            lst.append(random.uniform(-1, 1))
        W.append(lst)
    return np.array(W)


def generate_W3(hide_amount: int, amount_out: int):
    W = []
    for h in range(amount_out):
        lst = []
        for weight in range(hide_amount):
            lst.append(random.uniform(-1, 1))
        W.append(lst)
    return np.array(W)


def leaky_RELU(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] < 0:
                matrix[i][j] = matrix[i][j] * 0.01
    return matrix


def leaky_RELU_derivative(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] < 0:
                matrix[i][j] = 0.01
            else:
                matrix[i][j] = 1
    return matrix


def generate_T(hidden: int):
    W = []
    for h in range(hidden):
        lst = []
        for weight in range(1):
            lst.append(random.uniform(-1, 1))
        W.append(lst)
    return np.array(W)


def error_is(e, out):
    error = abs(e * e) + abs(out * out)
    error -= 2 * e * out
    error = error / 2
    return error


def update_W2(W2_old, ratio, out, e, hidden):
    error = out - e
    result = ratio * error
    # print(result, hidden)
    result = hidden * result
    # print("=======")
    # print(result)
    result = W2_old - result.T
    return result


def update_W1(W1_old, ratio, out, e, W2_new, F, enter):
    error = out - e
    result = ratio * error
    result = W2_new * result
    # print(F, enter.shape)
    buff = F @ enter.T
    result = result @ buff
    result = W1_old - result
    return result


def update_W3(W3_old, ratio, out, e, W2_new, F):
    error = out - e
    result = ratio * error
    result = W2_new * result
    buff = F * out
    result = result @ buff
    result = W3_old - result
    return result


def update_T(T_old, out, e):
    return T_old + (out - e)


