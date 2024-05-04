Scikit-optimize is a Python library for sequential model-based optimization, which provides a range of algorithms for hyperparameter tuning and optimization. Here are some examples of how to use scikit-optimize:

### 1. **Bayesian Optimization**

Bayesian optimization is a powerful method for hyperparameter tuning. Scikit-optimize provides several algorithms for Bayesian optimization, including `gp_minimize` and `forest_minimize`. These algorithms use a surrogate model to approximate the objective function and search for the optimal hyperparameters.

Example:
```python
from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer

def objective(x):
    # Define the objective function to minimize
    return x[0]**2 + x[1]**2

space = [Real(-5, 5, name='x1'), Real(-5, 5, name='x2')]
res_gp = gp_minimize(objective, space, n_calls=50, random_state=0)
print("Optimal parameters:", res_gp.x)
print("Optimal value:", res_gp.fun)
```
### 2. **Hyperparameter Search for Machine Learning Models**

Scikit-optimize provides a wrapper for scikit-learn's `GridSearchCV` and `RandomizedSearchCV` classes, called `BayesSearchCV`. This allows for Bayesian optimization of hyperparameters for machine learning models.

Example:
```python
from skopt import BayesSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

rfc = RandomForestClassifier(n_estimators=100)
bayes_search = BayesSearchCV(rfc, {'n_estimators': [10, 50, 100], 'max_depth': [None, 5, 10]}, cv=5, n_iter=10)
bayes_search.fit(X_train, y_train)
print("Best parameters:", bayes_search.best_params_)
print("Best score:", bayes_search.best_score_)
```
### 3. **Parallel Optimization**

Scikit-optimize provides tools for parallel optimization using Dask. This allows for distributed computing and can significantly speed up the optimization process.

Example:
```python
from skopt import gp_minimize
from skopt.space import Real
from dask.distributed import Client

client = Client(n_workers=4)

def objective(x):
    # Define the objective function to minimize
    return x[0]**2 + x[1]**2

space = [Real(-5, 5, name='x1'), Real(-5, 5, name='x2')]
res_gp = gp_minimize(objective, space, n_calls=50, random_state=0, client=client)
print("Optimal parameters:", res_gp.x)
print("Optimal value:", res_gp.fun)
```
### 4. **Adapter for Dask Parallel Tuning**

Scikit-optimize-adapter is a library built on top of scikit-optimize and Dask that provides an efficient and lightweight way to perform Bayesian optimization in parallel.

Example:
```python
from scikit_optimize_adapter import Adapter

def objective(x):
    # Define the objective function to minimize
    return x[0]**2 + x[1]**2

space = [Real(-5, 5, name='x1'), Real(-5, 5, name='x2')]
adapter = Adapter(objective, space, num_initial=5, num_iter=15)
res = adapter.run()
print("Optimal parameters:", res.x)
print("Optimal value:", res.fun)
```
These are just a few examples of how to use scikit-optimize for hyperparameter tuning and optimization. The library provides a range of algorithms and tools for sequential model-based optimization, making it a powerful tool for machine learning and optimization tasks.
