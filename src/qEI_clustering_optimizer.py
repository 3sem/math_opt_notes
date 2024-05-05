from skopt import Optimizer
from skopt.space import Real
from skopt.benchmarks import branin
from joblib import Parallel, delayed
import numpy as np
import sys

# Define the search space
space = [Real(-5.0, 10.0), Real(0.0, 15.0)]


# Define the objective function
def objective(params):
    x, y = params
    return branin((x, y))


# Create an optimizer instance
optimizer = Optimizer(space, random_state=1, base_estimator="gp")

# Define the clustering strategy
def clustering_strategy(optimizer, n_clusters=4):
    # Use k-means clustering to divide the search space into clusters
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(optimizer.space.transform(optimizer.Xi))

    # Optimize each cluster separately
    for cluster in np.unique(clusters):
        cluster_indices = np.where(clusters == cluster)[0][0]
        optimizer.tell(optimizer.Xi[cluster_indices], optimizer.yi[cluster_indices])

import time
# Run the optimization loop
ts = time.time()
for i in range(50):
    print("Opt loop teration", i)
    # Ask for the next point to evaluate
    x_next = optimizer.ask(n_points=4)

    # Evaluate the objective function
    # for x_item in x_next:
    optimizer.tell(x_next, Parallel(n_jobs=4)(delayed(objective)(v) for v in x_next))
        #y_next = objective(x_item)
    # Tell the optimizer the result
        #optimizer.tell(x_item, y_next)

    # Use the clustering strategy to optimize each cluster
    clustering_strategy(optimizer)
    print("Intermediate result:", min(optimizer.yi))
    print("Time elapsed:", time.time() - ts, "sec")
# Print the optimized parameters
print("Final result:", min(optimizer.yi), "at point" , optimizer.x)