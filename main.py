from benchmarks.experiment import run_experiment


def main(
    runs=100,
    population=10,
    iterations=50,
    dim=30,
    seed=42,
    output_dir="results",
):
    return run_experiment(
        runs=runs,
        population=population,
        iterations=iterations,
        dim=dim,
        seed=seed,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
