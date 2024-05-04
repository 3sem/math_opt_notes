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

### GPU

To parallelize a `gp_minimize` over several GPUs using scikit-optimize, you can use the `Optimizer` class directly, which provides an ask-and-tell interface[1][5]. This allows you to control the optimization loop and parallelize the evaluation of the objective function.

Here's an example of how you can modify the `gp_minimize` function to parallelize it over multiple GPUs:
```python
import os
import skopt
from skopt import Optimizer

# Define the search space
space = [Real(0, 10), Integer(2, 5)]

# Define the objective function
def objective(point, gpu_id='0'):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    # Evaluate the objective function on the specified GPU
    score = ...
    return score

# Create an Optimizer instance
opt = Optimizer(space)

# Define the number of GPUs and the number of initial points
n_gpus = 8
n_initial_points = 10

# Ask for the initial points
sampled_points = opt.ask(n_initial_points)

# Evaluate the objective function in parallel on each GPU
scores = []
for point, gpu_id in zip(sampled_points, range(n_gpus)):
    scores.append(objective(point, gpu_id=str(gpu_id)))

# Tell the optimizer about the results
opt.tell(sampled_points, scores)

# Continue the optimization loop
while True:
    # Ask for the next points to evaluate
    sampled_points = opt.ask(n_gpus)
    # Evaluate the objective function in parallel on each GPU
    scores = []
    for point, gpu_id in zip(sampled_points, range(n_gpus)):
        scores.append(objective(point, gpu_id=str(gpu_id)))
    # Tell the optimizer about the results
    opt.tell(sampled_points, scores)
```
In this example, we create an `Optimizer` instance and define the search space and the objective function. We then ask for the initial points to evaluate, and evaluate the objective function in parallel on each GPU using the `objective` function. We tell the optimizer about the results, and continue the optimization loop by asking for the next points to evaluate and evaluating them in parallel on each GPU.

Note that you'll need to modify the `objective` function to evaluate the objective function on the specified GPU, and you may need to use a library like `multiprocessing` or `dask` to parallelize the evaluation of the objective function[4].

Alternatively, you can use the `scikit-optimize-adapter` library, which provides a Dask parallelized Bayesian optimization toolbox[4]. This library allows you to parallelize the optimization process using Dask, which can be useful if you have a large number of GPUs available.

It's also worth noting that scikit-learn itself does not utilize GPUs, but you can use libraries like TensorFlow or PyTorch to build machine learning models that can utilize GPUs[3].
