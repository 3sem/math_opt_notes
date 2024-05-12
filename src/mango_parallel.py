from concurrent.futures import ThreadPoolExecutor # thread pool usage
import time
import math

from mango.tuner import Tuner
from scipy.stats import uniform
import random

# define the num of parallel workers
n_jobs = 4

# objfun raise an error when parameter x is <= 0
def objfun(params):
    x = params['x']
    if x <= 0:
        raise ValueError()
    return math.log(x)
# Obj_parallel uses concurrent.futures to parallelize the execution of
# objfun and handles the failed evaluation
def obj_parallel(params_list):
    futures = []
    params_evaluated = []
    results = []

    # here we are use thread executor which is ideal of I/O bound tasks
    # we can also use the ProcessPoolExecutor depending on the use case
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        for params in params_list:
            future = (params, executor.submit(objfun, params))
            futures.append(future)

        for params, future in futures:
            try:
                result = future.result()
                params_evaluated.append(params)
                results.append(result)
            except ValueError:
                print(f"Value Error raised for {params}")

    return params_evaluated, results

param_dict = {
    'x': uniform(-2, 10),
}

if __name__ == '__main__':
    tuner = Tuner(param_dict, obj_parallel, {'batch_size': n_jobs, 'num_iteration': 10})
    results = tuner.maximize()

    print('best parameters:',results['best_params'])
    print('best objective:',results['best_objective'])