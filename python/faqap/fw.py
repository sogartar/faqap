import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.optimize import OptimizeResult
from faqap.misc import random_doubly_stochastic_matrix
from faqap.permutation import (
    permutation_matrix,
    project_doubly_stochastic_matrix_onto_permutations,
)


# P is a permutation_matrix
def objective_with_mat(D, F, P):
    A = P @ D @ P.transpose()
    return np.tensordot(F, A, axes=2)


def objective(D, F, permutation):
    P = permutation_matrix(permutation, dtype=D.dtype)
    return objective_with_mat(D=D, F=F, P=P)


class SearchOriginGenerator:
    def __init__(self, n, dtype):
        self.n = n
        self.dtype = dtype
        self.center = np.full((n, n), 1.0 / n, dtype)

    def __call__(self):
        return 0.5 * (self.center + random_doubly_stochastic_matrix(self.n, self.dtype))


# Projects a gradient in matrix form onto the set of
# permutation matrices.
class TaylorExpansionMinimizer:
    def __call__(self, gradient):
        permutation = linear_sum_assignment(gradient)[1]
        perm_mat = permutation_matrix(permutation, gradient.dtype)
        return perm_mat


# Computes f(P) = <F, PDP^T> and its gradient.
# Note that f(P) is derived from |F + PDP^T|^2
# with terms dropped that are independent of P.
class Qap:
    def __init__(self, D, F):
        self.D = D
        self.F = F

    def __call__(self, P):
        return objective_with_mat(D=self.D, F=self.F, P=P)

    def gradient(self, P):
        return self.F.transpose() @ P @ self.D + self.F @ P @ self.D.transpose()


class LinearCombinationMinimizer:
    def __init__(self, D, F):
        self.D = D
        self.F = F
        self.qap = Qap(D, F)

    def __call__(self, X, Y):
        YmX = Y - X
        YmXD = YmX @ self.D
        A = YmXD @ YmX.transpose()
        a = np.tensordot(self.F, A, axes=2)

        if a < 0:
            obj_X = self.qap(X)
            obj_Y = self.qap(Y)
            return (0, X, obj_X) if obj_X < obj_Y else (1, Y, obj_Y)
        if a == 0:
            return (0, X, self.qap(X))

        B = YmXD @ X.transpose() + X @ self.D @ YmX.transpose()
        b = np.tensordot(self.F, B, axes=2)
        alpha = np.clip(-b / (2 * a), 0, 1)
        Z = alpha * YmX + X
        return (alpha, Z, self.qap(Z))


def minimize_relaxed(
    D, F, projector, x0_generator, count=1, maxiter=None, tol=1e-5, verbose=True
):
    qap = Qap(D, F)
    res = None
    linear_comb_opt = LinearCombinationMinimizer(D, F)

    j = 0
    for i in range(count):
        x = x0_generator()
        fun = np.finfo(D.dtype).max
        while True:
            grad = qap.gradient(x)
            y = projector(grad)
            (_, x_new, fun_new) = linear_comb_opt(x, y)

            if (
                (maxiter is not None and maxiter <= j)
                or np.linalg.norm(x - x_new) < tol
                or abs(fun - fun_new) < tol
            ):

                if res is None or fun_new < res.fun:
                    if res is None:
                        res = OptimizeResult()
                    res.x = x_new
                    res.fun = fun_new
                break

            x = x_new
            fun = fun_new
            j = j + 1

        if maxiter is not None and maxiter <= j:
            break

        if verbose and i % (np.maximum(1, count // 100)) == 0:
            print(
                "Frak-Wolfe QP progress = %.2f%%. Objective = %.3f."
                % (100.0 * (i + 1) / count, res.fun)
            )

    return res


# Minimizes f(P) = <F, PDP^T>, over P, which is a permutation matrix.
# <., .> is the Frobenius inner product.
# Returns a scipy.optimize.OptimizeResult object with members fun and x.
# x is the argument that minimizes f and fun is f(x).
# the permutation x is returned in line notation.
def minimize(D, F, descents_count=None):
    n = len(D)
    if descents_count is None:
        descents_count = n
    relaxed_sol = minimize_relaxed(
        D,
        F,
        projector=TaylorExpansionMinimizer(),
        x0_generator=SearchOriginGenerator(n, D.dtype),
        count=descents_count,
    )

    res = OptimizeResult()
    res.x = project_doubly_stochastic_matrix_onto_permutations(relaxed_sol.x)
    res.fun = objective(D, F, res.x)
    return res
