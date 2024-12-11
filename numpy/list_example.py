
def main(n):

    print(f'{n} elemement list')

    list_1 = []
    list_2 = []

    for i in range(n):
        list_1.append(i)

    for i in list_1:
        list_2.append(i * i)

    return list_2


if __name__ == "__main__":
    from sys import argv
    main(int(argv[1]))
