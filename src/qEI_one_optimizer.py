from skopt import Optimizer
from skopt.space import Real
from joblib import Parallel, delayed
from skopt.benchmarks import branin
import time
import sys


# Define the number of iterations and parallel trials
NUM_ITERATIONS = 10
NUM_PARALLEL_TRIALS = 4


# Perform the optimization
def DefaultOptimizing(num_iterations, num_parallel_trials, func,
                      dim=[Real(-5.0, 10.0), Real(0.0, 15.0)], verbose=True, strategy='cl_mean'):
    # Define the hyperparameter search space
    dimensions = dim
    # Create an Optimizer instance
    optimizer = Optimizer(dimensions, random_state=1, base_estimator="gp")
    if verbose:
        print("Default multi-point optimizer")
    all_scores_and_params = []
    for i in range(num_iterations):
        # Get a list of points in hyperparameter space to evaluate
        hyperparam_vals = optimizer.ask(n_points=num_parallel_trials, strategy=strategy)
        # Evaluate the points in parallel
        scores = Parallel(n_jobs=num_parallel_trials)(delayed(func)(v) for v in hyperparam_vals)
        # Update the optimizer with the results
        optimizer.tell(hyperparam_vals, scores)

        # Store the results
        all_scores_and_params.extend(zip(hyperparam_vals, scores))
        if verbose:
            print("On iter", i, "Optimizer's y'x are:", optimizer.yi)
# Print the best score found
    return optimizer, all_scores_and_params


if __name__ == '__main__':

    if sys.argv[1] == "t_eval":
        print("DefaultOptimizing: time evaluation mode")
        verbose = False
        try:
            num = int(sys.argv[2])
        except:
            num = 1
    else:
        num = 1
        print("DefaultOptimizing: one-pass verbose mode")
        verbose = True

    def run_strategies(strategy, num_rep, n_iter, num_threads, func, verbose):
        print("Running for strategy", strategy, num_rep, "times:")
        results = list()
        dt = list()
        for i in range(num_rep):
            st = time.time()
            optimizer, all_results = DefaultOptimizing(n_iter, num_threads, func, verbose=verbose)
            et = time.time()
            dt.append(et-st)
            # for these setting it seems that cl_mean and cl_min are better than cl_max
            results.append(min(optimizer.yi))
            print(i, ": Result:", results[-1])
        print("Time evaluation:", "Mean %f; " % (sum(dt) / num_rep), "Min %f ;" % min(dt), "Max %f ;" % max(dt))
        print("Results:", [(i, res) for i, res in enumerate(results)])
        if verbose:
            print(all_results)

    for st in ["cl_min", "cl_mean", "cl_max"]:
        run_strategies(strategy=st,
                       num_rep=num,
                       n_iter=NUM_ITERATIONS,
                       num_threads=NUM_PARALLEL_TRIALS,
                       func=branin,
                       verbose=verbose)
