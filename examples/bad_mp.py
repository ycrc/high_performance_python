import multiprocessing as mp
from math import sqrt
import numpy as np
import os

n_procs = int(os.getenv('SLURM_CPUS_ON_NODE'))

a = np.random.random(size=10000000)

with mp.Pool(n_procs) as pool:
    pool.map(np.sqrt, a)

