import numpy as np


def POSSolver(
    lb,
    ub,
    fn_obj,
    num_particles=5,
    iterations=10,
    c1=2.0,
    c2=2.0,
    dim=2,
    w_start=0.9,
    w_end=0.4,
    seed=None,
):
    if dim < 1:
        raise ValueError("dim must be at least 1")
    if num_particles < 1:
        raise ValueError("num_particles must be at least 1")
    if iterations < 1:
        raise ValueError("iterations must be at least 1")

    lb = np.broadcast_to(np.asarray(lb, dtype=float), (dim,)).copy()
    ub = np.broadcast_to(np.asarray(ub, dtype=float), (dim,)).copy()
    if not np.all(np.isfinite(lb)) or not np.all(np.isfinite(ub)):
        raise ValueError("Bounds must be finite")
    if np.any(lb >= ub):
        raise ValueError("Each lower bound must be smaller than its upper bound")

    rng = np.random.default_rng(seed)
    pos = rng.uniform(lb, ub, size=(num_particles, dim))
    velocities = np.zeros((num_particles, dim))

    pbest = pos.copy()
    pbest_values = np.array([fn_obj(particle) for particle in pos], dtype=float)

    best_index = np.argmin(pbest_values)
    gbest = pbest[best_index].copy()
    gbest_value = float(pbest_values[best_index])

    for iteration in range(iterations):
        r1 = rng.random((num_particles, dim))
        r2 = rng.random((num_particles, dim))

        progress = iteration / max(1, iterations - 1)
        inertia = w_start + progress * (w_end - w_start)
        velocities = (
            inertia * velocities
            + c1 * r1 * (pbest - pos)
            + c2 * r2 * (gbest - pos)
        )

        pos = np.clip(pos + velocities, lb, ub)
        values = np.array([fn_obj(particle) for particle in pos], dtype=float)

        improved = values < pbest_values
        pbest[improved] = pos[improved]
        pbest_values[improved] = values[improved]

        best_index = np.argmin(pbest_values)
        if pbest_values[best_index] < gbest_value:
            gbest = pbest[best_index].copy()
            gbest_value = float(pbest_values[best_index])

    return {
        "best position": gbest,
        "best value": gbest_value,
    }
