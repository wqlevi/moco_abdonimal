import jax
from functools import cache, partial
from timeit import timeit
import sys
sys.setrecursionlimit(10_000)


@cache
def factorial_cache(n):
    return n * factorial(n-1) if n else 1


def factorial(n):
    return n * factorial(n-1) if n else 1



if __name__ == '__main__':
    n = 1_000
    loop = 10_000
    rst = timeit('factorial_cache(n)',
                 globals=globals(),
                 number=loop) 
    print('cache runtime: {}'.format(rst/loop))

    rst = timeit('factorial(n)',
                 globals=globals(),
                 number=loop) 
    print('origin runtime: {}'.format(rst/loop))
