# Import libraries
import pandas as pd
import cvxpy as cp
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Define functions for training and testing soft-margin linear SVM
def svm_train_primal(data_train, label_train, regularisation_para_C):
    N, D = data_train.shape

    w = cp.Variable(D)
    b = cp.Variable()
    xi = cp.Variable(N)

    hinge_loss = cp.sum(cp.maximum(0, 1 - cp.multiply(label_train, data_train @ w + b)))
    regularisation_term = 0.5 * cp.norm(w, 'fro') ** 2
    soft_margin_term = regularisation_para_C * cp.sum(xi)
    
    objective = cp.Minimize(hinge_loss + regularisation_term + soft_margin_term)
    
    constraints = [xi >= 0, xi >= 1 - cp.multiply(label_train, data_train @ w + b)]
    
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)

    if prob.status == cp.OPTIMAL:
        svm_model = {
            'w': w.value,
            'b': b.value,
        }
        return svm_model
    else:
        raise Exception("SVM training failed")

def svm_predict_primal(data_test, label_test, svm_model):
    w = svm_model['w']
    b = svm_model['b']
    
    # Make predictions on the test data
    prediction = np.sign(data_test @ w + b)

    # Calculate test accuracy
    test_accuracy = accuracy_score(label_test, prediction)
    
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

    if prob.status == cp.OPTIMAL:
        svm_model = {
            'alpha': alpha.value,
        }
        return svm_model
    else:
        raise Exception("SVM training failed")
    
def get_primal_solution_from_dual(data, labels, a, C):
    # Find the indices of support vectors (0 < a_i < C)
    support_vector_indices = np.where((a > 0) & (a < C))[0]

    # checks for support vectors
    if len(support_vector_indices) == 0:
        raise Exception("No support vectors found")
    
    # initialize b*
    b_star = 0.0
    
    # Compute w* by summing over support vectors
    w_star = np.zeros(data.shape[1])

    for sv_index in support_vector_indices:
        w_star += a[sv_index] * labels[sv_index] * data[sv_index]
        b_star += labels[sv_index] - np.dot(w_star, data[sv_index])
    
    # average b* over support vectors
    b_star /= len(support_vector_indices)
    
    return w_star, b_star

def find_sv(data, labels, w_primal, b_primal):
    sv = []
    
    for i in range(len(data)):
        margin_condition = 1 - labels[i] * (np.dot(w_primal, data[i]) + b_primal)
        
        if margin_condition <= 0:
            sv.append(data[i])
    
    return np.array(sv)

def find_support_vectors_dual(data, labels, alpha, C):
    support_vector_indices = np.where((alpha > 0) & (alpha < C))[0]
    support_vectors = data[support_vector_indices]
    support_vector_labels = labels[support_vector_indices]
    return support_vectors, support_vector_labels

# Load the train and test datasets
train_data = pd.read_csv(r'path/train.csv') # load training data
test_data = pd.read_csv(r'path/test.csv') # load testing data

# Separate features (X) and labels (y) for training and test datasets
X_train = train_data.iloc[:4000, 1:].values
label_train = train_data.iloc[:4000, 0].values
X_val = train_data.iloc[4000:, 1:].values
label_val = train_data.iloc[4000:, 0].values
X_test = test_data.iloc[:, 1:].values
label_test = test_data.iloc[:, 0].values

# Scale the data
scaler = StandardScaler()
data_train = scaler.fit_transform(X_train)
data_val = scaler.transform(X_val)
data_test = scaler.transform(X_test)

# Set the regularization parameter C to 100
regularisation_para_C = 100

# Train the SVM model on the training set
svm_primal = svm_train_primal(data_train, label_train, regularisation_para_C)

# Test the SVM model on the test set and calculate test accuracy
test_accuracy = svm_predict_primal(data_test, label_test, svm_primal)
b = svm_primal['b']
w_sum = np.sum(svm_primal['w'])

svm_dual = svm_train_dual(data_train, label_train, regularisation_para_C)

# Calculate the sum of all dimensions of optimal alpha
sum_alpha = np.sum(svm_dual['alpha'])

w_star, b_star = get_primal_solution_from_dual(data_train, label_train, svm_dual['alpha'], regularisation_para_C)
support_vectorsa = find_sv(data_train, label_train, w_star, b_star)
support_vectors, support_vector_labels = find_support_vectors_dual(data_train, label_train, svm_dual['alpha'], regularisation_para_C)

# Define the range of C values to search within
C_values = [2 ** i for i in range(-10, 11)]

# Initialize variables to store the best C and its corresponding validation accuracy
best_C = None
best_validation_accuracy = 0.0

# Iterate through each C value and evaluate on the validation set
for C in C_values:
    # Train an SVM model using the dual formulation with the current C
    svm_model = svm_train_primal(data_train, label_train, C)
    
    # Test the SVM model on the validation set
    validation_accuracy = svm_predict_primal(data_val, label_val, svm_model)
    
    # Check if the current C resulted in a higher validation accuracy
    if validation_accuracy > best_validation_accuracy:
        best_C = C
        best_validation_accuracy = validation_accuracy

# Print the results
print ("b primal:", b)
b_dual = b_star
print("b dual:", b_dual)
print("Solution of b:", b)
print("Sum of all dimensions of w:", w_sum)
print("Test Accuracy:", test_accuracy)
print("Sum of all dimensions of optimal alpha:", sum_alpha)
print("w*:", w_star)
print("b*:", b_star)
print("Support Vectors:", support_vectorsa)
print("Support Vectors:", support_vectors)
print("Support Vector Labels:", support_vector_labels)
print("Number of support vectors (primal):", len(support_vectors))
# Report the best C and its corresponding validation accuracy
print("Best C:", best_C)
print("Validation Accuracy with Best C:", best_validation_accuracy)