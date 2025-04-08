# import libraries
import pandas as pd
import numpy as np
import cvxpy as cp
from sklearn.preprocessing import StandardScaler

def svm_train_dual(data_train, label_train, regularisation_para_C):
    N, num_features = data_train.shape

    # Define the optimization variables
    alpha = cp.Variable(N)
    
    # Define the SVM dual problem objective function
    kernel_matrix = np.dot(data_train, data_train.T)  # Compute the kernel matrix
    objective = cp.Maximize(cp.sum(alpha) - (1/2) * cp.quad_form(alpha * label_train, kernel_matrix))
    
    # Define constraints
    constraints = [0 <= alpha, alpha <= regularisation_para_C / N, cp.sum(alpha * label_train) == 0]
    
    # Create the optimization problem
    problem = cp.Problem(objective, constraints)
    
    # Solve the problem
    problem.solve()
    
    # Optimal alpha values
    alpha_optimal = alpha.value
    
    return alpha_optimal

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

# Call the SVM training function
svm_model = svm_train_dual(data_train, label_train, regularisation_para_C)

# Calculate the sum of all dimensions of the optimal alpha
sum_of_alpha = np.sum(svm_model)

print("Sum of all dimensions of optimal alpha:", sum_of_alpha)