import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.optimize import OptimizeResult
from faqap.misc import random_doubly_stochastic_matrix
from faqap.permutation import (
    permutation_matrix,
    project_doubly_stochastic_matrix_onto_permutations,
)
from faqap.torch import has_torch

if has_torch:
    import torch
    from faqap.torch import (
        TorchifiedSearchOriginGenerator,
        TorchifiedProjector,
        torch_dtype_to_numpy,
    )


# P is a permutation_matrix
def objective_with_mat(D, F, P):
    A = P @ D @ P.T
    if has_torch and isinstance(D, torch.Tensor):
        return torch.tensordot(F, A, dims=2).cpu()
    else:
        return np.tensordot(F, A, axes=2)


def objective(D, F, permutation):
    is_torch = has_torch and isinstance(D, torch.Tensor)
    if is_torch:
        numpy_dtype = torch_dtype_to_numpy[D.dtype]
    else:
        numpy_dtype = D.dtype
    P = permutation_matrix(permutation, dtype=numpy_dtype)
    if is_torch:
        P = torch.as_tensor(P).to(D.device)
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
        return self.F.T @ P @ self.D + self.F @ P @ self.D.T


class LinearCombinationMinimizer:
    def __init__(self, D, F):
        self.D = D
        self.F = F
        self.qap = Qap(D, F)

    def __call__(self, X, Y):
        is_torch = has_torch and isinstance(self.D, torch.Tensor)

        YmX = Y - X
        YmXD = YmX @ self.D
        A = YmXD @ YmX.T
        if is_torch:
            a = torch.tensordot(self.F, A, dims=2).cpu()
        else:
            a = np.tensordot(self.F, A, axes=2)

        if a < 0:
            obj_X = self.qap(X)
            obj_Y = self.qap(Y)
            return (0, X, obj_X) if obj_X < obj_Y else (1, Y, obj_Y)
        if a == 0:
            return (0, X, self.qap(X))

        B = YmXD @ X.T + X @ self.D @ YmX.T
        if is_torch:
            b = torch.tensordot(self.F, B, dims=2).cpu()
        else:
            b = np.tensordot(self.F, B, axes=2)
        alpha = np.clip(-b / (2 * a), 0, 1)
        Z = alpha * YmX + X
        return (alpha, Z, self.qap(Z))


def minimize_relaxed(
    D, F, projector, x0_generator, count=1, maxiter=None, tol=1e-5, verbose=False
):
    qap = Qap(D, F)
    res = None
    linear_comb_opt = LinearCombinationMinimizer(D, F)

    is_torch = has_torch and isinstance(D, torch.Tensor)
    if is_torch:
        numpy_dtype = torch_dtype_to_numpy[D.dtype]
    else:
        numpy_dtype = D.dtype

    j = 0
    for i in range(count):
        x = x0_generator()
        fun = np.finfo(numpy_dtype).max
        while True:
            grad = qap.gradient(x)
            y = projector(grad)
            (_, x_new, fun_new) = linear_comb_opt(x, y)

            if (maxiter is not None and maxiter <= j) or abs(fun - fun_new) < tol:

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

        if (verbose >= 3 and i % (np.maximum(1, count // 100)) == 0) or (
            verbose >= 2 and i % (np.maximum(10, count // 10)) == 0
        ):
            print(
                "Frak-Wolfe QP progress = %.2f%%. Objective = %.3f."
                % (100.0 * (i + 1) / count, res.fun)
            )

    return res


def minimize(
    D, F, x0_generator=None, descents_count=1, maxiter=None, tol=1e-5, verbose=0
):
    """
    Minimizes f(P) = <F, PDP^T>, over the set of permutation matrices.
    <., .> is the Frobenius inner product.
    This implementation uses the Frank–Wolfe algorithm.

    Parameters
    ----------
    D, F : square numpy matrices of the same size.
        By convention D is the distance and F is the flow in the factory assignment
        problem.
    x0_generator : generator for initial search points. It is a callable,
        that returns doubly stochastic matrices. The default generator picks random
        points (C + R)/2, where C is the center of the Birkhoff polytope and
        R is random matrix from it.
    descents_count : number of searches to perform from an initial point.
    maxiter : The maximum number of descent steps to perform. If None,
        there is no limit.
    tol : tolerance for the decrease of the objective. If the objective decreases
        with less than tol in one descent step, this local search is terminated.
    verbose : When True, prints results during the search.

    Returns
    -------
    scipy.optimize.OptimizeResult object with members fun and x.
    x is the argument that minimizes f and fun is f(x).
    the permutation x is returned in line notation.
    """
    n = len(D)

    is_torch = has_torch and isinstance(D, torch.Tensor)
    if is_torch:
        numpy_dtype = torch_dtype_to_numpy[D.dtype]
    else:
        numpy_dtype = D.dtype
    if x0_generator is None:
        x0_generator = SearchOriginGenerator(n, numpy_dtype)
        if is_torch:
            x0_generator = TorchifiedSearchOriginGenerator(
                x0_generator, device=D.device
            )
    projector = TaylorExpansionMinimizer()
    if is_torch:
        projector = TorchifiedProjector(projector)

    relaxed_sol = minimize_relaxed(
        D,
        F,
        projector=projector,
        x0_generator=x0_generator,
        count=descents_count,
        maxiter=maxiter,
        tol=tol,
        verbose=verbose,
    )

    res = OptimizeResult()
    if is_torch:
        res.x = project_doubly_stochastic_matrix_onto_permutations(relaxed_sol.x.cpu())
    else:
        res.x = project_doubly_stochastic_matrix_onto_permutations(relaxed_sol.x)
    res.fun = objective(D, F, res.x)
    if verbose >= 1:
        print("Frak-Wolfe QP objective = %.3f." % (res.fun))
    return res
