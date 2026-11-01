import numpy as np


def _penalty(x, threshold, scale=100.0, power=4):
    outside = np.maximum(np.abs(x) - threshold, 0.0)
    return scale * np.sum(outside ** power)


def _shekel(x, count):
    centers = np.array([
        [4.0, 4.0, 4.0, 4.0],
        [1.0, 1.0, 1.0, 1.0],
        [8.0, 8.0, 8.0, 8.0],
        [6.0, 6.0, 6.0, 6.0],
        [3.0, 7.0, 3.0, 7.0],
        [2.0, 9.0, 2.0, 9.0],
        [5.0, 5.0, 3.0, 3.0],
        [8.0, 1.0, 8.0, 1.0],
        [6.0, 2.0, 6.0, 2.0],
        [7.0, 3.6, 7.0, 3.6],
    ])
    constants = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])
    distances = np.sum((x - centers[:count]) ** 2, axis=1)
    return -np.sum(1.0 / (distances + constants[:count]))


def build_benchmarks(dim, seed=None):
    benchmarks = []
    quartic_rng = np.random.default_rng(seed)

    def add(name, fn, lb, ub, dimensions=dim, minimum=0.0):
        index = len(benchmarks) + 1
        if index <= 7:
            group = "unimodal"
        elif index <= 13:
            group = "multimodal"
        else:
            group = "fixed_dimension_multimodal"

        benchmarks.append({
            "name": name,
            "group": group,
            "fn": fn,
            "lb": lb,
            "ub": ub,
            "dim": dimensions,
            "minimum": minimum,
        })

    add("F1 Sphere", lambda x: np.sum(x ** 2), -100.0, 100.0)
    add(
        "F2 Schwefel 2.22",
        lambda x: np.sum(np.abs(x)) + np.prod(np.abs(x)),
        -10.0,
        10.0,
    )
    add(
        "F3 Schwefel 1.2",
        lambda x: np.sum(np.cumsum(x) ** 2),
        -100.0,
        100.0,
    )
    add("F4 Schwefel 2.21", lambda x: np.max(np.abs(x)), -100.0, 100.0)
    add(
        "F5 Rosenbrock",
        lambda x: np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1.0) ** 2),
        -30.0,
        30.0,
    )
    add("F6 Step", lambda x: np.sum(np.floor(x + 0.5) ** 2), -100.0, 100.0)
    def quartic(x):
        return np.sum(np.arange(1, x.size + 1) * x ** 4) + quartic_rng.uniform(0.0, 1.0)

    add("F7 Quartic", quartic, -128.0, 128.0)
    add(
        "F8 Schwefel 2.26",
        lambda x: -np.sum(x * np.sin(np.sqrt(np.abs(x)))),
        -500.0,
        500.0,
        minimum=-418.9829 * dim,
    )
    add(
        "F9 Rastrigin",
        lambda x: 10.0 * x.size + np.sum(x ** 2 - 10.0 * np.cos(2.0 * np.pi * x)),
        -5.12,
        5.12,
    )
    add(
        "F10 Ackley",
        lambda x: (
            -20.0 * np.exp(-0.2 * np.sqrt(np.mean(x ** 2)))
            - np.exp(np.mean(np.cos(2.0 * np.pi * x)))
            + 20.0
            + np.e
        ),
        -32.0,
        32.0,
    )
    add(
        "F11 Griewank",
        lambda x: 1.0 + np.sum(x ** 2) / 4000.0 - np.prod(
            np.cos(x / np.sqrt(np.arange(1, x.size + 1)))
        ),
        -600.0,
        600.0,
    )

    def penalized_1(x):
        y = 1.0 + (x + 1.0) / 4.0
        value = 10.0 * np.sin(np.pi * y[0]) ** 2
        value += np.sum((y[:-1] - 1.0) ** 2 * (1.0 + 10.0 * np.sin(np.pi * y[1:]) ** 2))
        value += (y[-1] - 1.0) ** 2
        return np.pi * value / x.size + _penalty(x, 10.0)

    add("F12 Penalized 1", penalized_1, -50.0, 50.0)

    def penalized_2(x):
        value = np.sin(3.0 * np.pi * x[0]) ** 2
        value += np.sum((x[:-1] - 1.0) ** 2 * (1.0 + np.sin(3.0 * np.pi * x[1:]) ** 2))
        value += (x[-1] - 1.0) ** 2 * (1.0 + np.sin(2.0 * np.pi * x[-1]) ** 2)
        return 0.1 * value + _penalty(x, 5.0)

    add("F13 Penalized 2", penalized_2, -50.0, 50.0)

    grid = np.arange(-32.0, 33.0, 16.0)
    foxholes = np.array([(a, b) for a in grid for b in grid])

    def shekel_foxholes(x):
        distances = np.sum((foxholes - x) ** 6, axis=1)
        denominators = np.arange(1.0, 26.0) + distances
        return 1.0 / (0.002 + np.sum(1.0 / denominators))

    add("F14 Foxholes", shekel_foxholes, -65.536, 65.536, 2, 0.9980038)

    kowalik_a = np.array([
        0.1957, 0.1947, 0.1735, 0.16, 0.0844, 0.0627,
        0.0456, 0.0342, 0.0323, 0.0235, 0.0246,
    ])
    kowalik_b = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0])

    def kowalik(x):
        numerator = x[0] * (kowalik_b ** 2 + kowalik_b * x[1])
        denominator = kowalik_b ** 2 + kowalik_b * x[2] + x[3]
        return np.sum((kowalik_a - numerator / denominator) ** 2)

    add("F15 Kowalik", kowalik, -5.0, 5.0, 4, 0.0003075)
    add(
        "F16 Six-hump Camel",
        lambda x: (
            (4.0 - 2.1 * x[0] ** 2 + x[0] ** 4 / 3.0) * x[0] ** 2
            + x[0] * x[1]
            + (-4.0 + 4.0 * x[1] ** 2) * x[1] ** 2
        ),
        -5.0,
        5.0,
        2,
        -1.0316285,
    )

    def branin(x):
        a = 1.0
        b = 5.1 / (4.0 * np.pi ** 2)
        c = 5.0 / np.pi
        r = 6.0
        s = 10.0
        t = 1.0 / (8.0 * np.pi)
        return a * (x[1] - b * x[0] ** 2 + c * x[0] - r) ** 2 + s * (1.0 - t) * np.cos(x[0]) + s

    add("F17 Branin", branin, -5.0, 5.0, 2, 0.397887)

    def goldstein_price(x):
        first = 1.0 + (x[0] + x[1] + 1.0) ** 2 * (
            19.0 - 14.0 * x[0] + 3.0 * x[0] ** 2
            - 14.0 * x[1] + 6.0 * x[0] * x[1] + 3.0 * x[1] ** 2
        )
        second = 30.0 + (2.0 * x[0] - 3.0 * x[1]) ** 2 * (
            18.0 - 32.0 * x[0] + 12.0 * x[0] ** 2
            + 48.0 * x[1] - 36.0 * x[0] * x[1] + 27.0 * x[1] ** 2
        )
        return first * second

    add("F18 Goldstein-Price", goldstein_price, -2.0, 2.0, 2, 3.0)

    hartmann_alpha = np.array([1.0, 1.2, 3.0, 3.2])
    hartmann3_a = np.array([
        [3.0, 10.0, 30.0],
        [0.1, 10.0, 35.0],
        [3.0, 10.0, 30.0],
        [0.1, 10.0, 35.0],
    ])
    hartmann3_p = 1e-4 * np.array([
        [3689.0, 1170.0, 2673.0],
        [4699.0, 4387.0, 7470.0],
        [1091.0, 8732.0, 5547.0],
        [381.0, 5743.0, 8828.0],
    ])

    def hartmann3(x):
        return -np.sum(hartmann_alpha * np.exp(-np.sum(hartmann3_a * (x - hartmann3_p) ** 2, axis=1)))

    add("F19 Hartmann 3D", hartmann3, -1.0, 2.0, 3, -3.86278)

    hartmann6_a = np.array([
        [10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
        [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
        [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
        [17.0, 8.0, 0.05, 10.0, 0.1, 14.0],
    ])
    hartmann6_p = np.array([
        [0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
        [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
        [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
        [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381],
    ])

    def hartmann6(x):
        return -np.sum(hartmann_alpha * np.exp(-np.sum(hartmann6_a * (x - hartmann6_p) ** 2, axis=1)))

    add("F20 Hartmann 6D", hartmann6, 0.0, 1.0, 6, -3.32237)
    add("F21 Shekel 5", lambda x: _shekel(x, 5), 0.0, 10.0, 4, -10.1532)
    add("F22 Shekel 7", lambda x: _shekel(x, 7), 0.0, 10.0, 4, -10.4029)
    add("F23 Shekel 10", lambda x: _shekel(x, 10), 0.0, 10.0, 4, -10.5364)

    return benchmarks