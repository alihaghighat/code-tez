import torch
import numpy as np


def generate_initial_parameters(param_optimizer):
    """
    Generate initial hyperparameters for optimization.
    """
    random_input = torch.rand(1, 10)
    output = param_optimizer(random_input)

    lookback = int(output[0][0] * 20 + 20)  # Range: 20 to 40
    delay = int(output[0][1] * -4 - 1)  # Range: -1 to -5
    n_top = int(output[0][2] * 2 + 4)  # Range: 2 to 4
    return lookback, delay, n_top


def mutate_parameters(best_params, scale=0.5):
    """
    Mutate hyperparameters based on the best parameters.
    """
    lookback, delay, n_top = best_params
    lookback = max(20, min(40, int(lookback + np.random.normal(0, scale * lookback))))
    delay = max(-5, min(-1, int(delay + np.random.normal(0, scale * abs(delay)))))
    n_top = max(2, min(6, int(n_top + np.random.normal(0, scale * n_top))))
    return lookback, delay, n_top
