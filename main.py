from benchmarks.functions import build_benchmarks
from utils.plot import _make_qualitative_plots

import numpy as np
from pathlib import Path
from utils import _save_csv, _run

def main(runs=100, population=10, iterations=50, dim=30, seed=42):

    # Benchmarks 
    """
        Criterials to evaluate:
        1. search history
        2. trayectory of the first agent
        3. average fitness
        4. convergence curve 
    """
    benchmarks = build_benchmarks(dim)

    statistics = []
    mean_scores = {}
    output_path = Path('.')
    for id, benchmark in enumerate(benchmarks):
        mean_scores[benchmark["name"]] = {}
        for algorithm in ["WO", "POS", "ACO"]:
            scores = []
            runtimes = []
            for run_id in range(runs):
                run_seed = seed + id * runs + run_id
                result, elapsed, _ = _run(
                    algorithm,
                    benchmark,
                    population,
                    iterations,
                    run_seed,
                )
                scores.append(float(result["best value"]))
                runtimes.append(elapsed)

            mean_scores[benchmark["name"]][algorithm] = float(np.mean(scores))
            statistics.append({
                "benchmark": benchmark["name"],
                "group": benchmark["group"],
                "dimension": benchmark["dim"],
                "algorithm": algorithm,
                "runs": runs,
                "population": population,
                "iterations": iterations,
                "best": float(np.min(scores)),
                "worst": float(np.max(scores)),
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores, ddof=1)) if runs > 1 else 0.0,
                "median_runtime_seconds": float(np.median(runtimes)),
            })
            
        # Qualitative analysis
        if benchmark["name"].split(" ")[0] in [f'F{i}' for i in [1, 9, 14]]:
            diagnostic, _, _ = _run(
                "WO",
                benchmark,
                population,
                iterations,
                seed + id * runs,
                collect_history=True,
            )
            _make_qualitative_plots(
                benchmark,
                diagnostic,
                output_path / "qualitative" / benchmark["name"],
            )
    # Compare results
    comparisons = []
    for algorithm in ["POS", "ACO"]:
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
        comparisons.append({"algorithm": algorithm, "wins_vs_wo": wins, "ties_vs_wo": ties, "losses_vs_wo": losses})

    # Save results
    _save_csv(output_path / "statistics.csv", statistics)
    _save_csv(output_path / "comparison_with_wo.csv", comparisons)

    print(f"Comparación: {runs} corridas, población {population}, {iterations} iteraciones")
    print(f"Resultados cuantitativos: {output_path / 'statistics.csv'}")
    print(f"Comparación de medias contra WO: {output_path / 'comparison_with_wo.csv'}")

if __name__ == "__main__":
    """
        F1–F7 unimodals
        F8–F13 multimodals
        F14–F23 fixed-dimension multimodal
    """
    main()