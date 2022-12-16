import math


def fibonacci_series(steps):
    result = [0, 1]
    for i in range(steps):
        result.append(result[len(result) - 1] + result[len(result) - 2])
    return result


def factorial_series(amount):
    result = []
    for i in range(amount):
        # print(i)
        i += 1
        result.append(math.factorial(i))
    return result


def periodic_series(amount):
    result = []
    number = -1
    for i in range(amount):
        i += 1
        result.append(pow(number, i))
    return result


def power_function(amount):
    result = []
    for i in range(amount):
        i += 1
        result.append(pow(i, 2))
    return result
