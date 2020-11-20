import numpy as np
from scipy.optimize import linear_sum_assignment


def identity_permutation(size, dtype=None):
    return np.arange(size, dtype=dtype)


def random_permutation(size, dtype=None):
    res = identity_permutation(size, dtype)
    np.random.shuffle(res)
    return res


# apply p1, then p2
def compose_permutations(p2, p1):
    dtype = p1.dtype if hasattr(p1, "dtype") else np.int32
    res = np.empty(len(p1), dtype=dtype)
    for i in range(len(p1)):
        res[i] = p2[p1[i]]
    return res


def inverse_permutation(permutation):
    res = np.empty(len(permutation), dtype=np.int)
    for i in range(len(permutation)):
        res[permutation[i]] = i
    return res


def permutation_matrix(permutation, dtype=int):
    res = np.zeros((len(permutation), len(permutation)), dtype)
    for i in range(len(permutation)):
        res[i, permutation[i]] = 1
    return res


# Projects the doubly stochastic matrix P onto the set of permutations.
# Returns the permutation in line notation.
def project_doubly_stochastic_matrix_onto_permutations(P):
    return linear_sum_assignment(P, maximize=True)[1]
