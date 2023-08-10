# work out paths
import os
import sys

# this is intended to be used by pytest from parent directory

up = os.path.normpath(os.path.join(os.getcwd(), "src"))
sys.path.append(up)
sys.path.append(os.getcwd())

import numpy as np
from maxmf.minimize import minimize
import pytest

from min_funcs import funclist

n = 5


@pytest.mark.parametrize("fn, gconv, crit", funclist)
def test_minimize(fn, gconv, crit):
    init = 0.5 * (np.random.rand(n) - 0.5)
    result = minimize(fn, init, maxiters=500, gconv=gconv)
    print(result)
    if result["converged"]:
        assert np.linalg.norm(result["x"]) < crit * n, f"min not correct: {np.max(np.abs(result['x']))}"
    else:
        assert False, "not converged"
