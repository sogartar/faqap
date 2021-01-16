#!/usr/bin/env python3

from faqap import minimize
from unittest import TestCase, main
from faqap.fw import objective
import numpy as np
import torch
import itertools
import random


class FaqapTorchDeviceAgnostic:
    def __init__(self, device):
        self.device = device

    def init_random_seed(self):
        random.seed(782347079)
        np.random.seed(123456789)
        torch.manual_seed(987654321)

    def test_against_exhaustive_list(self):
        D = torch.tensor(
            np.array(
                [
                    [63, 18, -4, 98],
                    [93, 54, 89, -91],
                    [-69, -82, 85, -85],
                    [-56, -78, 39, 4],
                ],
                np.float64,
            )
        ).to(self.device)
        F = torch.tensor(
            np.array(
                [
                    [36, -62, -15, 55],
                    [41, -82, 57, 39],
                    [50, 12, -49, -61],
                    [-69, 70, -40, -6],
                ],
                np.float64,
            )
        ).to(self.device)

        objective_permutation_pairs = [
            (objective(D=D, F=F, permutation=perm), perm)
            for perm in itertools.permutations([0, 1, 2, 3])
        ]
        expected = min(objective_permutation_pairs, key=lambda x: x[0])

        solution = minimize(D=D, F=F, descents_count=1)

        np.testing.assert_almost_equal(expected[0], solution.fun, decimal=5)
        np.testing.assert_array_equal(expected[1], solution.x)

    def test_equivalence_between_numpy_and_torch(self):
        n = 64
        D = torch.rand(n, n, dtype=torch.float32)
        F = torch.rand(n, n, dtype=torch.float32)

        self.init_random_seed()
        solution_numpy = minimize(D=D.numpy(), F=F.numpy(), descents_count=10)
        self.init_random_seed()
        solution_torch = minimize(
            D=D.to(self.device), F=F.to(self.device), descents_count=10
        )

        np.testing.assert_array_equal(solution_numpy.x, solution_torch.x)
        np.testing.assert_allclose(solution_numpy.fun, solution_torch.fun, rtol=1e-05)


class FaqapTorchCpuTest(TestCase, FaqapTorchDeviceAgnostic):
    def __init__(self, *args, **kwargs):
        TestCase.__init__(self, *args, **kwargs)
        FaqapTorchDeviceAgnostic.__init__(self, torch.device("cpu"))

    def setUp(self):
        TestCase.setUp(self)
        self.init_random_seed()


class FaqapTorchCudaTest(TestCase, FaqapTorchDeviceAgnostic):
    def __init__(self, *args, **kwargs):
        TestCase.__init__(self, *args, **kwargs)
        FaqapTorchDeviceAgnostic.__init__(self, torch.device("cuda"))

    def setUp(self):
        TestCase.setUp(self)
        self.init_random_seed()


if __name__ == "__main__":
    main()
