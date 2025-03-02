import numpy as np
import math
import copy

# loading the training data into the model
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])


# cost function
def compute_cost(x, y, w, b):
    size = x.shape[0]
    cost = 0

    for i in range(size):
        f_wb = (x[i] * w) + b
        cost = cost + (f_wb - y[i])**2

    return cost / (2 * size)


def compute_gradient(x, y, w, b):
    size = x.shape[0]

    dj_dw = 0
    dj_db = 0

    for i in range(size):
        dj_dw += (((x[i] * w) + b) - y[i]) * x[i]
        dj_db += ((x[i] * w) + b) - y[i]

    return dj_dw/size, dj_db/size


# computing the gradient descent function
def gradient_descent(x, y, w_in, b_in, alpha, num_iters):
    # array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    p_history = []
    b = b_in
    w = w_in

    for i in range(num_iters):
        # calculating the gradient and update the parameters using gradient_function
        dj_dw, dj_db = compute_gradient(x, y, w, b)

        # updating Parameters using equation (3) above
        b = b - alpha * dj_db
        w = w - alpha * dj_dw

        if i < 100000:
            J_history.append(compute_cost(x, y, w, b))
            p_history.append([w, b])

        # displaying the ouputs every other interval
        if i % math.ceil(num_iters/10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")

    return w, b, J_history, p_history  # return w and J,w history for graphing


# initialize parameters
w_init = 0
b_init = 0
# some gradient descent settings
iterations = 10000

tmp_alpha = 0.1e-1
# run gradient descent
w_final, b_final, J_hist, p_hist = gradient_descent(x_train, y_train, w_init, b_init, tmp_alpha,
                                                    iterations)
print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")
