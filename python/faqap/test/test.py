#!/usr/bin/env python3

from faqap import minimize
from unittest import TestCase, main
from faqap.permutation import permutation_matrix
from faqap.test.permutation import inverse_permutation
from faqap.fw import objective
import numpy as np
import itertools


class FaqapTest(TestCase):
    def __init__(self, *args, **kwargs):
        super(FaqapTest, self).__init__(*args, **kwargs)

    def setUp(self):
        TestCase.setUp(self)
        np.random.seed(123456789)

    def test_inverse(self):
        F = np.array(
            [[1, 2, 3, 4], [2, 5, 6, 7], [3, 6, 8, 9], [4, 7, 9, 13]], np.float64
        )
        correct_permutation = [1, 3, 2, 0]
        P = permutation_matrix(inverse_permutation(correct_permutation))
        D = -P @ F @ P.transpose()
        solution_permutation = minimize(D=D, F=F, descents_count=1).x
        np.testing.assert_array_equal(correct_permutation, solution_permutation)

    def test_against_exhaustive_list(self):
        D = np.array(
            [
                [63, 18, -4, 98],
                [93, 54, 89, -91],
                [-69, -82, 85, -85],
                [-56, -78, 39, 4],
            ],
            np.float64,
        )
        F = np.array(
            [
                [36, -62, -15, 55],
                [41, -82, 57, 39],
                [50, 12, -49, -61],
                [-69, 70, -40, -6],
            ],
            np.float64,
        )

        objective_permutation_pairs = [
            (objective(D=D, F=F, permutation=perm), perm)
            for perm in itertools.permutations([0, 1, 2, 3])
        ]
        expected = min(objective_permutation_pairs, key=lambda x: x[0])

        solution = minimize(D=D, F=F, descents_count=1)

        np.testing.assert_almost_equal(expected[0], solution.fun, decimal=5)
        np.testing.assert_array_equal(expected[1], solution.x)

    def test_reverse_small(self):
        D = np.array([[0, 0, -3], [0, -2, 0], [-1, 0, 0]], dtype=np.float64)
        F = np.array([[0, 0, +1], [0, +2, 0], [+3, 0, 0]], dtype=np.float64)

        expected_permutation = [2, 1, 0]
        expected_objective = objective(D=D, F=F, permutation=expected_permutation)

        solution = minimize(D=D, F=F, descents_count=3)

        np.testing.assert_almost_equal(expected_objective, solution.fun, decimal=5)
        np.testing.assert_array_equal(expected_permutation, solution.x)

    def test_reverse_big(self):
        D = np.array(
            [[0, 0, 0, -4], [0, 0, -3, 0], [0, -2, 0, 0], [-1, 0, 0, 0]],
            dtype=np.float64,
        )
        F = np.array(
            [[0, 0, 0, +1], [0, 0, +2, 0], [0, +3, 0, 0], [+4, 0, 0, 0]],
            dtype=np.float64,
        )

        expected_permutation = [3, 2, 1, 0]
        expected_objective = objective(D=D, F=F, permutation=expected_permutation)

        solution = minimize(D=D, F=F, descents_count=1)

        np.testing.assert_almost_equal(expected_objective, solution.fun, decimal=5)
        np.testing.assert_array_equal(expected_permutation, solution.x)


if __name__ == "__main__":
    main()
