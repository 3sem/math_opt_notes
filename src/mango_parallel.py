
import numpy as np
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

# Load iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Define the KNN classifier
clf = KNeighborsClassifier()

# Define the hyperparameter search space
param_space = {
    'n_neighbors': range(1, 50),
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'leaf_size': range(10, 50),
    'p': [1, 2]
}

# Define the objective function to optimize
def objective(hyperparameters):
    clf.set_params(**hyperparameters)
    scores = cross_val_score(clf, X, y, cv=5)
    return np.mean(scores)

# Initialize Mango
mango = Mango(objective, param_space, num_iterations=50)

# Run optimization
mango.run()

# Get results
best_hyperparameters = mango.best_hyperparameters
best_score = mango.best_score
print(f"Best hyperparameters: {best_hyperparameters}")
print(f"Best score: {best_score}")