# import libraries
import pandas as pd
import numpy as np
import cvxpy as cp
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def svm_train_primal(data_train, label_train, regularisation_para_C):
    N, D = data_train.shape

    # define variables
    w = cp.Variable(D)
    b = cp.Variable()
    xi = cp.Variable(N)

    # define constraints
    constraints = [cp.multiply(label_train, (data_train @ w + b)) >= 1 - xi, xi >= 0]

    # define objective
    objective = cp.Minimize(0.5 * cp.norm(w, 2) ** 2 + (regularisation_para_C / N) * cp.sum(xi))

    # create and solve SVM primal problem
    prob = cp.Problem(objective, constraints)
    prob.solve()

    # extract solutions
    svm_model = {
        'w': w.value,
        'b': b.value
    }

    return svm_model

def svm_predict_primal(data_test, label_test, svm_model):
    w = svm_model['w']
    b = svm_model['b']

    # predict using the SVM model
    predictions = np.sign(data_test @ w + b)

    test_accuracy = accuracy_score(label_test, predictions)

    return test_accuracy

def svm_train_dual(data_train, label_train, regularisation_para_C):
    N, D = data_train.shape

    alpha = cp.Variable(N)
    
    # Define the objective function (linear expression)
    objective = cp.Maximize(cp.sum(alpha) - 0.5 * cp.norm(cp.multiply(label_train, alpha).T @ data_train, 'fro')**2)
    
    # Define the constraints
    constraints = [alpha >= 0, alpha <= regularisation_para_C / N, cp.sum(cp.multiply(label_train, alpha)) == 0]
    
    prob = cp.Problem(objective, constraints)
    prob.solve()

    svm_model = {
        'a': alpha.value
    }
    
    return svm_model

def get_primal_solution_lagrangian(data_train, label_train, a_star):
    # Find support vectors with non-zero Lagrange multipliers
    support_vector_indices = np.where((a_star > 1e-5) & (a_star < regularisation_para_C - 1e-5))[0]

    # Compute 'w*' as the sum of 'a_i * y_i * x_i' over support vectors
    w_star = np.sum(a_star[support_vector_indices] * label_train[support_vector_indices][:, np.newaxis] * data_train[support_vector_indices], axis=0)

    # Compute 'b*' as the average of 'y_i - <w*, x_i>' over support vectors
    b_star = np.mean(label_train[support_vector_indices] - np.dot(data_train[support_vector_indices], w_star))

    return w_star, b_star

# load datasets
train_data = pd.read_csv(r'path/train.csv') # load training data
test_data = pd.read_csv(r'path/test.csv') # load testing data

# separate the data into training and test sets
# first 4000 samples for training
X_train = train_data.iloc[:4000, 1:].values
label_train = train_data.iloc[:4000, 0].values
X_test = test_data.iloc[:, 1:].values
label_test = test_data.iloc[:, 0].values

# standardize the features
scaler = StandardScaler()
data_train = scaler.fit_transform(X_train)
data_test = scaler.transform(X_test)

regularisation_para_C = 100

# train SVM model
svm_modelp = svm_train_primal(data_train, label_train, regularisation_para_C)

# test SVM model and calculate test accuracy
test_accuracy = svm_predict_primal(data_test, label_test, svm_modelp)

svm_modeld = svm_train_dual(data_train, label_train, regularisation_para_C)

w_star, b_star = get_primal_solution_lagrangian(data_train, label_train, svm_modeld['a'])

# report solution of b and sum of all dimensions of w
print("Solution of b:", svm_modelp['b'])
print("Sum of all dimensions of w:", np.sum(svm_modelp['w']))
print("Test Accuracy:", test_accuracy)
print("Sum of all dimensions of optimal a:", np.sum(svm_modeld['a']))
print("w*:", w_star)
print("b*:", b_star)