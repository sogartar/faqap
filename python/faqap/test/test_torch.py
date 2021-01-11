#!/usr/bin/env python3

from faqap import minimize
from unittest import TestCase, main
from faqap.fw import objective
import numpy as np
import torch
import itertools
import random


class FaqapTorchTest(TestCase):
    def __init__(self, *args, **kwargs):
        super(FaqapTorchTest, self).__init__(*args, **kwargs)

    def setUp(self):
        TestCase.setUp(self)
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
        )
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
        )

        objective_permutation_pairs = [
            (objective(D=D, F=F, permutation=perm), perm)
            for perm in itertools.permutations([0, 1, 2, 3])
        ]
        expected = min(objective_permutation_pairs, key=lambda x: x[0])

        solution = minimize(D=D, F=F, descents_count=1)

        np.testing.assert_almost_equal(expected[0], solution.fun, decimal=5)
        np.testing.assert_array_equal(expected[1], solution.x)


if __name__ == "__main__":
    main()
