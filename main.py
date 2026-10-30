import numpy as np
from algorithms.wo import WOSolver

if __name__ == "__main__":
    def sphere(x):
        return np.sum(x**2)

    result = WOSolver(
        lb=-5.12,
        ub=5.12,
        fn_obj=sphere,
        num_walruses=10,
        T=50,
        dim=2,
        seed=42,
    )
    print(result)