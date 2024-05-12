import torch
import torch.nn as nn
import torch.optim as optim
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.acquisition import UpperConfidenceBound
from botorch.utils import get_best_candidates
from botorch.test_functions import Branin

# Define the objective function
def branin(x):
    x1, x2 = x
    return (x2 - (5.1/(4*np.pi**2))*x1**2 + (5/np.pi)*x1 - 6)**2 + 10*(1 - 1/(8*np.pi))*np.cos(x1) + 10

# Define the bounds for the objective function
bounds = torch.tensor([[0, 15], [0, 15]])

# Define the number of parallel evaluations
num_parallel = 4

# Define the acquisition function
acqf = UpperConfidenceBound(model=SingleTaskGP, beta=0.1)

# Define the optimization problem
problem = optimize_acqf(
    acq_function=acqf,
    bounds=bounds,
    num_candidates=num_parallel,
    num_iterations=10,
    q=1,
    random_state=0
)

# Run the optimization
results = optimize_acqf(problem)

# Get the best candidate
best_candidate = get_best_candidates(results, num_best=1)[0]

# Evaluate the objective function at the best candidate
best_value = branin(best_candidate)

print(f"Best candidate: {best_candidate}")
print(f"Best value: {best_value}")
