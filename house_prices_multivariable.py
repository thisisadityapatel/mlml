import numpy as np
import math
import copy
np.set_printoptions(precision=2)  # reduced display precision on numpy arrays


# training data
x_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])


def compute_prediction(x, w, b):  # computing the prediction
    prediction = np.dot(x, w) + b
    return prediction


def compute_cost(x, y, w, b):  # computing the cost
    m = x.shape[0]
    cost = 0

    for i in range(m):
        c = ((np.dot(x[i], w) + b) - y[i]) ** 2
        cost += c
    cost = cost / (2 * m)
    return cost


def compute_gradient(x, y, w, b):  # computing the gradient dj_dw and dj_db
    m = x.shape[0]
    x_m = x.shape[1]
    dj_dw = np.zeros((x_m, ))
    dj_db = 0
    for i in range(m):
        temp_fun = (np.dot(x[i], w) + b) - y[i]
        dj_db += temp_fun
        for j in range(x_m):
            dj_dw[j] += temp_fun * x[i, j]

    dj_db = dj_db / m
    dj_dw = dj_dw / m
    return dj_dw, dj_db


# computing the gradient descent function
def compute_gradient_descent(x, y, w_initial, b_initial, iterations, alpha):
    w = copy.deepcopy(w_initial)
    b = b_initial
    J_tracker = []

    for i in range(iterations):
        dj_dw, dj_db = compute_gradient(x, y, w, b)  # None

        # Updating the new w and b parameters
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        # Save cost J at each iteration
        if i < 100000:      # for safety
            J_tracker.append(compute_cost(x, y, w, b))

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(iterations / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_tracker[-1]:8.2f}   ")

    return w, b, J_tracker


# initialize training model
initial_w = np.zeros((x_train.shape[1], ))
initial_b = 0.

# some gradient descent settings
iterations = 1000
alpha = 5.0e-7

# run gradient descent
w_final, b_final, J_hist = compute_gradient_descent(
    x_train, y_train, initial_w, initial_b, iterations, alpha)

print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")

# compring the predictions with the training data into the model
m, _ = x_train.shape
for i in range(m):
    print(
        f"prediction: {np.dot(x_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")
