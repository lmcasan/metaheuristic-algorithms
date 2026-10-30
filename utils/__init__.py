from algorithms.aco import ACOSolver
from algorithms.pos import POSSolver
from algorithms.wo import WOSolver

import time
import csv

def _run(algorithm, benchmark, population, iterations, seed, collect_history=False):
    evaluations = 0

    def objective(position):
        nonlocal evaluations
        evaluations += 1
        return float(benchmark["fn"](position))

    parameters = {
        "lb": benchmark["lb"],
        "ub": benchmark["ub"],
        "fn_obj": objective,
        "dim": benchmark["dim"],
        "seed": seed,
    }

    start = time.perf_counter()
    if algorithm == "WO":
        result = WOSolver(
            **parameters,
            num_walruses=population,
            T=iterations,
            return_history=collect_history,
        )
    elif algorithm == "POS":
        result = POSSolver(
            **parameters,
            num_particles=population,
            iterations=iterations,
        )
    else:
        result = ACOSolver(
            **parameters,
            num_ants=population,
            archive_size=population,
            iterations=iterations,
        )
    elapsed = time.perf_counter() - start

    return result, elapsed, evaluations


def _save_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)