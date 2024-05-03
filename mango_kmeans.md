Here is an example of using Mango for Bayesian optimization with k-means clustering for parallel search:
``` { .py }
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import celery

# Define the search space for KNN classifier's hyperparameters
param_space = dict(n_neighbors=range(1, 50), algorithm=['auto', 'ball_tree', 'kd_tree', 'brute'])

# Define the objective function that trains the KNN classifier for a given value of hyperparameters
def objective(params):
    X, y = datasets.load_breast_cancer(return_X_y=True)
    clf = KNeighborsClassifier(**params)
    scores = cross_val_score(clf, X, y, cv=5)
    return scores.mean()

# Create a Celery app and define a remote task for the objective function
app = celery.Celery('knn_tuning', backend='rpc://')
@app.task
def remote_objective(params):
    return objective(params)

# Define a custom scheduler with 4 jobs
from mango import scheduler
@scheduler.custom(n_jobs=4)
def parallel_objective(params_batch):
    jobs = [remote_objective.s(params) for params in params_batch]
    return [job.get() for job in jobs]

# Create a Tuner instance with the search space, objective function, and number of iterations
from mango import Tuner
tuner = Tuner(param_space, parallel_objective, {'num_iteration': 30})

# Run the optimization
results = tuner.maximize()

# Print the best parameters and accuracy
print('Best parameters:', results['best_params'])
print('Best accuracy:', results['best_objective'])
```
This example uses the `mango` library to perform Bayesian optimization with k-means clustering for parallel search. The `objective` function trains a KNN classifier with different hyperparameters and evaluates its performance using cross-validation. The `parallel_objective` function is a custom scheduler that uses Celery to distribute the evaluation of the objective function across 4 jobs. The `Tuner` instance is created with the search space, objective function, and number of iterations, and the optimization is run using the `maximize` method. The best parameters and accuracy are printed at the end.

Note that this example assumes you have Celery installed and configured on your system. You may need to modify the `app` and `remote_objective` definitions to match your Celery setup. Additionally, you can adjust the number of jobs and iterations to suit your specific use case.
