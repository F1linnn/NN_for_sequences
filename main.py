from sequences import *
from W_actions import *

# 1 3
test1 = [[1, 2, 3],
        [2, 3, 4],
        [3, 4, 5],
        [4, 5, 6], ]
etalon1 = [4, 5, 6, 7]

test2 = [[0, 1, 1],
        [1, 1, 2],
        [1, 2, 3],
        [2, 3, 5], ]
etalon2 = [2, 3, 5, 8]

# test = [[1, 2, 3],
#         [2, 3, 4],
#         [3, 4, 5],
#         [4, 5, 6], ]
# etalon = [4, 5, 6, 7]


def start(amount_out, iter, ratio, test, etalon):
    error = 100000
    context = np.array([[0]])
    W1 = generate_W1(4, len(test[0]))  # 4 - число нейронов на скрытом слое
    W2 = generate_W2(amount_out, 4)
    W3 = generate_W3(amount_out, 4)
    T = generate_T(4)  # Т - смещение размером n, где n - это число нейронов на скрытом слое
    counter = 0
    while counter < iter:
        for i in range(len(test)):
            counter += 1
            # if i == 0:
            #     context = np.array([[0]])
            enter = np.array([test[i]])
            enter = enter.T
            context = W3 @ context
            # print(context)
            hidden = W1 @ enter
            hidden = hidden + context
            hidden = hidden - T
            before_relu = hidden
            hidden = leaky_RELU(hidden)
            context = W2 @ hidden
            error = abs(error_is(etalon[i], context[0][0]))
            print(f"_______Поколение №{counter}_____________")
            print(f"Последовательность: {test[i]} , эталон {etalon[i]}")
            print(f"Error= {error}, число от сети: {context[0][0]}")
            # print(f"Our number is {context[0][0]}")
            W2 = update_W2(W2, ratio, context[0][0], etalon[i], hidden)
            W1 = update_W1(W1, ratio, context[0][0], etalon[i], W2, leaky_RELU_derivative(before_relu), enter)
            W3 = update_W3(W3, ratio, context[0][0], etalon[i], W2, leaky_RELU_derivative(before_relu))
            T = update_T(T, context[0][0], etalon[i])


if __name__ == "__main__":
    # amount_out = int(input("Введите необходимое кол-во угадываемых чисел: "))
    # iter = int(input("Введите число итераций: "))
    # ratio = float(input("Введите коэффициент обучения: "))
    start(1, 500, 0.00001,test1,etalon1)  # угадываем следующее число последовательности
