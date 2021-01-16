# Fast Approximate Quadratic Assignment Problem Solver

This is a Python implementation of an algorithm for approximately solving quadratic
assignment problems described in

Joshua T. Vogelstein and John M. Conroy and Vince Lyzinski and Louis J. Podrazik and
Steven G. Kratzer and Eric T. Harley and Donniell E. Fishkind and
R. Jacob Vogelstein and Carey E. Priebe
(2012) Fast Approximate Quadratic Programming for Large (Brain) Graph Matching.
[arXiv:1112.5507](https://arxiv.org/abs/1112.5507).

It solves

min<sub>𝑃∈𝒫</sub><𝐹, 𝑃𝐷𝑃<sup>𝖳</sup>>

where 𝐷, 𝐹 ∈ ℝ<sup>𝑛×𝑛</sup>, 𝒫 is the set of 𝑛×𝑛 permutation matrices
and <., .> denotes the Frobenius inner product.

The implementation employs the
[Frank–Wolfe algorithm](https://en.wikipedia.org/wiki/Frank%E2%80%93Wolfe_algorithm).


## Example
```python
import numpy as np
from faqap import minimize

# Make runs deterministic, descent origins are chosen randomly by default.
np.random.seed(123456789)

D = np.array(
    [
        [0, 0, 0, -4],
        [0, 0, -3, 0],
        [0, -2, 0, 0],
        [-1, 0, 0, 0]
    ],
    dtype=np.float64
)
F = np.array(
    [
        [0, 0, 0, +1],
        [0, 0, +2, 0],
        [0, +3, 0, 0],
        [+4, 0, 0, 0]
    ],
    dtype=np.float64
)

solution_permutation = minimize(D=D, F=F, descents_count=1).x

# Expected is the permutation reversing elements.
print("solution permutation =", solution_permutation)

```

Output
```
solution permutation = [3 2 1 0]
```


### GPU Support
GPU support is enabled through Torch. It is an optional dependency.
In order to use the GPU you must pass Torch tensors that are on the CUDA device.
If you pass CPU tensors the GPU will not be used.

Note that linear sum assignment, which is a part of the algorithm, is done on
the CPU though SciPy.
On a system with GPU GeForce RTX 2080 SUPER and
CPU AMD Ryzen Threadripper 2920X (single thread at 3.5 - 4.3 GHz)
for a float32, 128 sized problem linear sum assignment takes ~60% of the
execution time.
It may be possible to move that part on the GPU as well, but currently
there are no good off-the-shelf GPU implementations for that.
It is also unclear if there will be any significant speedup.


## Installation
```
pip install faqap
```

## Dependencies
* Python (>=3.5)
* NumPy (>=1.10)
* SciPy (>=1.4)
* Torch (optional)
