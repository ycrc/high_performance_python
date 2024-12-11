import numpy as np

def main(n):

    print(f'{n} elemement list')

    array_1 = np.arange(0,n)

    return np.pow(array_1, 2)


if __name__ == "__main__":
    from sys import argv
    main(int(argv[1]))
