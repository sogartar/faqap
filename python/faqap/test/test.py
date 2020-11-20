from faqap import minimize
from unittest import TestCase
from faqap.permutation import permutation_matrix, inverse_permutation
import numpy as np


class FaqapTest(TestCase):
    def __init__(self, *args, **kwargs):
        super(FaqapTest, self).__init__(*args, **kwargs)

    def setUp(self):
        TestCase.setUp(self)
        np.random.seed(123456789)

    def test_qap_fw(self):
        F = np.array(
            [[1, 2, 3, 4], [2, 5, 6, 7], [3, 6, 8, 9], [4, 7, 9, 13]], np.float64
        )
        correct_permutation = [1, 3, 2, 0]
        P = permutation_matrix(inverse_permutation(correct_permutation))
        D = P.transpose() @ F @ P
        solution_permutation = minimize(D, F, 1).x
        np.testing.assert_array_equal(correct_permutation, solution_permutation)
