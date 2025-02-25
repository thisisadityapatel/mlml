import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
np.set_printoptions(precision=2)


def load_house_data():
    data = np.loadtxt("_data.txt", delimiter=',', skiprows=1)
    X = data[:, :4]
    y = data[:, 4]
    print(X)
    return X, y


# LOADING DATA SET
X_train, y_train = load_house_data()


# SCALING NORMALIZING TRAINING DATA
scaler = StandardScaler()
# using the z-score normalization of the data instead of the casual mean-normalization
X_norm = scaler.fit_transform(X_train)


# CREATING AND FITTING THE REGRESSION MODEL
# with max 1000 iterations for initial testing purposes
sgdr = SGDRegressor(max_iter=1000)
sgdr.fit(X_norm, y_train)

# GETTING THE COFFECIENT ALGORITHM
b_norm = sgdr.intercept_   # b varibale F(X)
w_norm = sgdr.coef_        # w variabel of F(X)

# prediction of some data using the model
y_pred_sgd = sgdr.predict(X_norm)

# COMPARING THE TWO DATA
print("Prediction Data: ")
print(y_pred_sgd)
print("Training Data Ouput: ")
print(y_train)
