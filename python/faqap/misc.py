import numpy as np


def random_doubly_stochastic_matrix(n, dtype):
    x = np.random.rand(n, n).astype(dtype)
    rsum = np.zeros(n, dtype)
    csum = np.zeros(n, dtype)

    ones = np.ones(n, dtype)
    abs_tolerance = np.finfo(dtype).eps * 100
    while not np.allclose(rsum, ones, rtol=0, atol=abs_tolerance) or not np.allclose(
        csum, ones, rtol=0, atol=abs_tolerance
    ):
        x /= x.sum(0)
        x = x / x.sum(1)[:, np.newaxis]
        rsum = x.sum(1)
        csum = x.sum(0)

    return x
