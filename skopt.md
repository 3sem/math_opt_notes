### CPU (qEI in parallel with CL)

Here is an example of Bayesian multi-core optimization using scikit-optimize:

```
from skopt import Optimizer
from skopt.space import Real
from joblib import Parallel, delayed
from skopt.benchmarks import branin

# Define the hyperparameter search space
dimensions = [Real(-5.0, 10.0), Real(0.0, 15.0)]

# Create an Optimizer instance
optimizer = Optimizer(dimensions, random_state=1, base_estimator="gp")

# Define the number of iterations and parallel trials
NUM_ITERATIONS = 4
NUM_PARALLEL_TRIALS = 4

# Perform the optimization
all_scores_and_params = []
for i in range(NUM_ITERATIONS):
    # Get a list of points in hyperparameter space to evaluate
    hyperparam_vals = optimizer.ask(n_points=NUM_PARALLEL_TRIALS)
    
    # Evaluate the points in parallel
    scores = Parallel(n_jobs=NUM_PARALLEL_TRIALS)(delayed(branin)(v) for v in hyperparam_vals)
    
    # Update the optimizer with the results
    optimizer.tell(hyperparam_vals, scores)
    
    # Store the results
    all_scores_and_params.extend(zip(hyperparam_vals, scores))

# Print the best score found
print(min(optimizer.yi))
```

This example uses the `Optimizer` class from scikit-optimize to perform Bayesian optimization of a function (in this case, the `branin` function). 

The `ask` method is used to generate a list of hyperparameter values to evaluate, and the `tell` method is used to update the optimizer with the results. 

The `Parallel` class from joblib is used to evaluate the points in parallel across multiple CPU cores.

