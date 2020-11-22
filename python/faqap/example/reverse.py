import numpy as np
from faqap import minimize

# Make runs deterministic, descent origins are chosen randomly by default.
np.random.seed(123456789)

D = np.array(
    [[0, 0, 0, -4], [0, 0, -3, 0], [0, -2, 0, 0], [-1, 0, 0, 0]], dtype=np.float64
)
F = np.array(
    [[0, 0, 0, +1], [0, 0, +2, 0], [0, +3, 0, 0], [+4, 0, 0, 0]], dtype=np.float64
)

solution_permutation = minimize(D=D, F=F, descents_count=1).x

# Expected is the permutation reversing elements.
print("solution permutation =", solution_permutation)
