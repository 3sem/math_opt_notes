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

### Custom k-means points clustering example[horizontal scaling]
The skopt library provides several strategies for optimization, including clustering-based strategies. Here are some examples:

### 1. **Clustering-based optimization**

Skopt provides a clustering-based optimization strategy, which involves dividing the search space into clusters and optimizing each cluster separately[3]. This approach can be useful when the objective function has multiple local optima or when the search space is high-dimensional.

### 2. **K-means clustering**

K-means clustering is a popular unsupervised learning algorithm that can be used for clustering-based optimization[4]. In skopt, k-means clustering can be used to divide the search space into clusters, and then optimize each cluster separately.

### 3. **Batch Gaussian process bandits**

Batch Gaussian process bandits is another strategy used in skopt for parallelizing Bayesian optimization[3]. This approach involves using k-means clustering to divide the search space into clusters, and then using Gaussian process bandits to optimize each cluster separately.

### 4. **Hyperparameter tuning**

Skopt can be used for hyperparameter tuning of machine learning models[2]. This involves optimizing the hyperparameters of a model to achieve the best performance on a given dataset. Clustering-based strategies can be used to divide the hyperparameter space into clusters, and then optimize each cluster separately.

### 5. **Bayesian optimization**

Skopt provides a Bayesian optimization strategy, which involves using a probabilistic approach to optimize the objective function[5]. This approach can be used in conjunction with clustering-based strategies to optimize the objective function.

Here is an example code snippet that demonstrates how to use skopt for Bayesian optimization with a clustering-based strategy:
```python
from skopt import Optimizer
from skopt.space import Real
from skopt.benchmarks import branin

# Define the search space
space = [Real(-5.0, 10.0), Real(0.0, 15.0)]

# Define the objective function
def objective(params):
    x, y = params
    return branin(x, y)

# Create an optimizer instance
optimizer = Optimizer(space, base_estimator='gp', n_random_starts=10)

# Define the clustering strategy
def clustering_strategy(optimizer, n_clusters=5):
    # Use k-means clustering to divide the search space into clusters
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(optimizer.space.transform(optimizer.Xi))

    # Optimize each cluster separately
    for cluster in np.unique(clusters):
        cluster_indices = np.where(clusters == cluster)[0]
        optimizer.tell(optimizer.Xi[cluster_indices], optimizer.yi[cluster_indices])

# Run the optimization loop
for i in range(50):
    # Ask for the next point to evaluate
    x_next = optimizer.ask()

    # Evaluate the objective function
    y_next = objective(x_next)

    # Tell the optimizer the result
    optimizer.tell(x_next, y_next)

    # Use the clustering strategy to optimize each cluster
    clustering_strategy(optimizer)

# Print the optimized parameters
print(optimizer.x)
```
This code snippet demonstrates how to use skopt for Bayesian optimization with a clustering-based strategy. The `clustering_strategy` function is used to divide the search space into clusters using k-means clustering, and then optimize each cluster separately using the `Optimizer` instance.
