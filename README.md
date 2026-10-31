# Metaheuristic algorithms

The project compares the Walrus Optimizer (WO), Particle Swarm Optimization
(POS), and Ant Colony Optimization (ACO) on 23 benchmark functions.

## Install dependencies

```bash
python -m pip install -r requirements.txt
```

## Run the full experiment

```bash
python main.py
```

The default experiment uses 100 runs, a population of 10, and 50 iterations.
It writes per-run measurements to `results/run_results.csv`, aggregate
metrics to `results/statistics.csv`, win/tie/loss counts to
`results/comparison_with_wo.csv`, and WO diagnostic plots under
`results/qualitative/`.

Each algorithm receives the same benchmark, population, iteration count, and
run seed. All three perform one population of initial evaluations and one
population per iteration, so the objective-evaluation budget is
`population * (iterations + 1)` per run. Runtime can still vary by algorithm.

The benchmark suite is grouped as F1–F7 unimodal, F8–F13 multimodal, and
F14–F23 fixed-dimension multimodal. F7 uses a deterministic quartic variant
(without the random additive noise sometimes used in its classical definition)
so repeated objective evaluations remain reproducible.
