#!/usr/bin/env python3

from faqap import minimize
from unittest import TestCase, main
import numpy as np
import torch
import random
import timeit


class FaqapTorchBenchmark(TestCase):
    def __init__(self, *args, **kwargs):
        super(FaqapTorchBenchmark, self).__init__(*args, **kwargs)

    def init_random_seed(self):
        random.seed(782347079)
        np.random.seed(123456789)
        torch.manual_seed(987654321)

    def setUp(self):
        TestCase.setUp(self)
        self.init_random_seed()

    def test_benchmark_torch_cuda_128(self):
        n = 128
        print("Running torch cuda benchmark %dx%d" % (n, n))

        device = torch.device("cuda")
        D = torch.rand(n, n, dtype=torch.float32).to(device)
        F = torch.rand(n, n, dtype=torch.float32).to(device)

        # Force lazy init
        minimize(D=D, F=F)

        start = timeit.default_timer()
        minimize(D=D, F=F, descents_count=10)
        stop = timeit.default_timer()
        execution_time = stop - start
        print("Execution time (s) =", execution_time)

    def test_benchmark_torch_cuda_256(self):
        n = 256
        print("Running torch cuda benchmark %dx%d" % (n, n))

        device = torch.device("cuda")
        D = torch.rand(n, n, dtype=torch.float32).to(device)
        F = torch.rand(n, n, dtype=torch.float32).to(device)

        # Force lazy init
        minimize(D=D, F=F)

        start = timeit.default_timer()
        minimize(D=D, F=F, descents_count=10)
        stop = timeit.default_timer()
        execution_time = stop - start
        print("Execution time (s) =", execution_time)

    def test_benchmark_torch_cpu_128(self):
        n = 128
        print("Running torch cpu benchmark %dx%d" % (n, n))

        D = torch.rand(n, n, dtype=torch.float32)
        F = torch.rand(n, n, dtype=torch.float32)

        # Force lazy init
        minimize(D=D, F=F)

        start = timeit.default_timer()
        minimize(D=D, F=F, descents_count=10)
        stop = timeit.default_timer()
        execution_time = stop - start
        print("Execution time (s) =", execution_time)

    def test_benchmark_torch_cpu_256(self):
        n = 256
        print("Running torch cpu benchmark %dx%d" % (n, n))

        D = torch.rand(n, n, dtype=torch.float32)
        F = torch.rand(n, n, dtype=torch.float32)

        # Force lazy init
        minimize(D=D, F=F)

        start = timeit.default_timer()
        minimize(D=D, F=F, descents_count=10)
        stop = timeit.default_timer()
        execution_time = stop - start
        print("Execution time (s) =", execution_time)

    def test_benchmark_numpy_128(self):
        n = 128
        print("Running torch cuda benchmark %dx%d" % (n, n))

        D = torch.rand(n, n, dtype=torch.float32).numpy()
        F = torch.rand(n, n, dtype=torch.float32).numpy()

        # Force lazy init
        minimize(D=D, F=F)

        start = timeit.default_timer()
        minimize(D=D, F=F, descents_count=10)
        stop = timeit.default_timer()
        execution_time = stop - start
        print("Execution time (s) =", execution_time)

    def test_benchmark_numpy_256(self):
        n = 256
        print("Running torch cuda benchmark %dx%d" % (n, n))

        D = torch.rand(n, n, dtype=torch.float32).numpy()
        F = torch.rand(n, n, dtype=torch.float32).numpy()

        # Force lazy init
        minimize(D=D, F=F)

        start = timeit.default_timer()
        minimize(D=D, F=F, descents_count=10)
        stop = timeit.default_timer()
        execution_time = stop - start
        print("Execution time (s) =", execution_time)


if __name__ == "__main__":
    main()
