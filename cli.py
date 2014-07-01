import sys
from collage_maker import CollageMaker

def read_int():
    return int(read_str())


def read_str():
    return sys.stdin.readline().strip()


def main(args):
    N = read_int()
    data = [read_int() for _ in range(N)]
    ret = CollageMaker().compose(data)
    for r in ret:
        print r
    sys.stdout.flush()

if __name__ == '__main__':
    main(sys.argv)
