import numpy as np


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
