import numpy as np
import pandas as pd
from numba import jit, prange
import random

def basic_mc(n_samples = 1000000):
    import random

    # set random seed
    random.seed(42)

    x = []
    y = []

    for i in range(n_samples):
        x_tmp = random.random()
        y_tmp = random.random()

        x.append(x_tmp)
        y.append(y_tmp)

    r = []
    for i in range(len(x)):
        r_tmp = (x[i]**2 + y[i]**2)**0.5
        r.append(r_tmp)

    counter = 0
    for i in range(len(r)):
        if r[i] < 1:
            counter += 1

    return 4 * counter / n_samples

def numpy_mc(n_samples = 1000000):

    np.random.seed(42)

    x = np.random.random(size=n_samples)
    y = np.random.random(size=n_samples)

    r = np.power(np.power(x,2) + np.power(y,2), 0.5)

    counter = r[r<1].size

    return 4 * counter / n_samples

@jit(nopython=True)
def numba_mc(n_samples = 1000000):

    # set random seed
    random.seed(42)

    x = []
    y = []

    for i in range(n_samples):
        x_tmp = random.random()
        y_tmp = random.random()

        x.append(x_tmp)
        y.append(y_tmp)

    r = []
    for i in range(len(x)):
        r_tmp = (x[i]**2 + y[i]**2)**0.5
        r.append(r_tmp)

    counter = 0
    for i in range(len(r)):
        if r[i] < 1:
            counter += 1

    return 4 * counter / n_samples

@jit(nopython=True, parallel = True)
def numba_par_mc(n_samples = 1000000):

    # set random seed
    random.seed(42)

    x = np.zeros(n_samples)
    y = np.zeros(n_samples)

    for i in prange(n_samples):
        x_tmp = random.random()
        y_tmp = random.random()

        x[i] = x_tmp
        y[i] = y_tmp

    r = np.zeros(n_samples)
    for i in prange(len(x)):
        r_tmp = (x[i]**2 + y[i]**2)**0.5
        r[i] = r_tmp

    counter = 0
    for i in prange(len(r)):
        if r[i] < 1:
            counter += 1

    return 4 * counter / n_samples


@jit(nopython=True, parallel = True)
def numba_par_mc_loop(n_samples = 1000000):

    counter = 0
    for i in prange(n_samples):
        x = random.random()
        y = random.random()
        if x**2 + y**2 < 1:
            counter += 1
    return 4 * counter / n_samples

if __name__ == '__main__':

    from sys import argv
    import time
    import matplotlib.pyplot as plt

    if len(argv) == 1:
        n_samples = [1000, 10000, 100000, 1000000, 10000000, 100000000]
    else:
        n_samples = [int(x) for x in argv[1:]]

    res_basic = []
    res_numpy = []
    res_numba = []
    res_par_numba = []
    res_par_loop_numba = []

    for n in n_samples:
        print(f'n_samples: {n}')

        print('Basic MC: ', end='')
        tick = time.time()
        basic_mc(n)
        tock = time.time()
        res_basic.append(tock-tick)
        print(f'{tock-tick:.6f}s')

        print('Numpy MC: ', end='')
        tick = time.time()
        numpy_mc(n)
        tock = time.time()
        res_numpy.append(tock-tick)
        print(f'{tock-tick:.6f}s')

        print('Numba MC: ', end='')
        tick = time.time()
        numba_mc(n)
        tock = time.time()
        res_numba.append(tock-tick)
        print(f'{tock-tick:.6f}s')

        print('Parallel Numba MC: ', end='')
        tick = time.time()
        numba_par_mc(n)
        tock = time.time()
        res_par_numba.append(tock-tick)
        print(f'{tock-tick:.6f}s')

        print('Parallel Numba Loop MC: ', end='')
        tick = time.time()
        numba_par_mc_loop(n)
        tock = time.time()
        res_par_loop_numba.append(tock-tick)
        print(f'{tock-tick:.6f}s')

    f = plt.figure()
    plt.plot(n_samples, res_basic, label='basic')
    plt.plot(n_samples, res_numpy, label='numpy')
    plt.plot(n_samples, res_numba, label='numba')
    plt.plot(n_samples, res_par_numba, label='parallel numba')
    plt.plot(n_samples, res_par_loop_numba, label='parallel numba (2)')
    plt.xlabel('samples')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('runtime (s)')
    plt.legend()
    plt.savefig('mc.pdf')


