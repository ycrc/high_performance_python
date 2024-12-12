import numpy as np
import pandas as pd
from numba import jit, prange
import random

def numpy_mc(n_samples = 10000000, seed=42):

    np.random.seed(seed)

    x = np.random.random(size=n_samples)
    y = np.random.random(size=n_samples)

    r = np.power(np.power(x,2) + np.power(y,2), 0.5)

    counter = r[r<1].size

    return 4 * counter / n_samples

if __name__ == '__main__':

    from sys import argv
    import time
    import os

    seed = int(argv[1])

    pi_calc = numpy_mc(n_samples = 1000000000, seed=seed)

    print(f'{os.getenv("SLURM_ARRAY_TASK_ID")}: {pi_calc:.10f}')


