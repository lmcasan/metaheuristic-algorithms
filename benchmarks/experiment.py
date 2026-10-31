from pathlib import Path

import numpy as np

from .functions import build_benchmarks
from utils.experiment import run_algorithm, save_csv
from utils.plot import make_qualitative_plots


ALGORITHMS = ("WO", "POS", "ACO")
QUALITATIVE_FUNCTIONS = {"F1", "F9", "F14"}


def run_experiment(
    runs=100,
    population=10,
    iterations=50,
    dim=30,
    seed=42,
    output_dir="results",
):
    if runs < 1 or population < 3 or iterations < 1 or dim < 1:
        raise ValueError(
            "runs, iterations, and dim must be positive; population must be at least 3."
        )

    output = Path(output_dir)
    benchmarks = build_benchmarks(dim)
    expected_evaluations = population * (iterations + 1)
    run_results = []
    statistics = []
    mean_scores = {}

    for benchmark_index, benchmark in enumerate(benchmarks):
        benchmark_name = benchmark["name"]
        mean_scores[benchmark_name] = {}

        for algorithm in ALGORITHMS:
            scores = []
            runtimes = []
            evaluation_counts = []

            for run_index in range(runs):
                run_seed = seed + benchmark_index * runs + run_index
                result, elapsed, evaluations = run_algorithm(
                    algorithm,
                    benchmark,
                    population,
                    iterations,
                    run_seed,
                )
                if evaluations != expected_evaluations:
                    raise RuntimeError(
                        f"{algorithm} used {evaluations} objective evaluations; "
                        f"expected {expected_evaluations}."
                    )

                best_value = float(result["best value"])
                scores.append(best_value)
                runtimes.append(elapsed)
                evaluation_counts.append(evaluations)
                run_results.append({
                    "benchmark": benchmark_name,
                    "group": benchmark["group"],
                    "dimension": benchmark["dim"],
                    "algorithm": algorithm,
                    "run": run_index + 1,
                    "seed": run_seed,
                    "best_value": best_value,
                    "runtime_seconds": elapsed,
                    "objective_evaluations": evaluations,
                })

            mean_scores[benchmark_name][algorithm] = float(np.mean(scores))
            statistics.append({
                "benchmark": benchmark_name,
                "group": benchmark["group"],
                "dimension": benchmark["dim"],
                "algorithm": algorithm,
                "runs": runs,
                "population": population,
                "iterations": iterations,
                "objective_evaluations_per_run": evaluation_counts[0],
                "best": float(np.min(scores)),
                "worst": float(np.max(scores)),
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores, ddof=1)) if runs > 1 else 0.0,
                "median_runtime_seconds": float(np.median(runtimes)),
            })

        function_id = benchmark_name.split()[0]
        if function_id in QUALITATIVE_FUNCTIONS:
            diagnostic_seed = seed + benchmark_index * runs
            diagnostic, _, evaluations = run_algorithm(
                "WO",
                benchmark,
                population,
                iterations,
                diagnostic_seed,
                collect_history=True,
            )
            if evaluations != expected_evaluations:
                raise RuntimeError("WO diagnostic run used an unexpected evaluation budget.")
            make_qualitative_plots(
                benchmark,
                diagnostic,
                output / "qualitative" / function_id,
            )

    comparisons = []
    for algorithm in ALGORITHMS[1:]:
        wins = ties = losses = 0
        for benchmark in benchmarks:
            wo_mean = mean_scores[benchmark["name"]]["WO"]
            other_mean = mean_scores[benchmark["name"]][algorithm]
            if np.isclose(wo_mean, other_mean, rtol=1e-12, atol=1e-15):
                ties += 1
            elif other_mean < wo_mean:
                wins += 1
            else:
                losses += 1
        comparisons.append({
            "algorithm": algorithm,
            "benchmarks_compared": len(benchmarks),
            "wins_vs_wo": wins,
            "ties_vs_wo": ties,
            "losses_vs_wo": losses,
        })

    save_csv(output / "run_results.csv", run_results)
    save_csv(output / "statistics.csv", statistics)
    save_csv(output / "comparison_with_wo.csv", comparisons)

    print(f"Comparison: {runs} runs, population {population}, {iterations} iterations")
    print(f"Per-run results: {output / 'run_results.csv'}")
    print(f"Summary metrics: {output / 'statistics.csv'}")
    print(f"Mean-score comparison with WO: {output / 'comparison_with_wo.csv'}")
    print(f"WO qualitative plots: {output / 'qualitative'}")

    return {
        "run_results": run_results,
        "statistics": statistics,
        "comparisons": comparisons,
        "output_dir": output,
    }
