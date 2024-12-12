import multiprocessing as mp
from math import sqrt
import numpy as np
import os

n_procs = int(os.getenv('SLURM_CPUS_ON_NODE'))

a = np.random.random(size=1000000000)

np.sqrt(a)


