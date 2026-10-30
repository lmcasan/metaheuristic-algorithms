import numpy as np


def POSSolver(lb, ub, fn_obj, num_particles=5, iterations=10, c1=2.0, c2=2.0, dim=2, w_start=0.9, w_end=0.4, seed = 0.1):
    # Assignment of values
    c1 = c1
    c2 = c2
    lb = np.broadcast_to(lb, (dim,)).copy()
    ub = np.broadcast_to(ub, (dim,)).copy()

    # Generating initial positions
    rng = np.random.default_rng(None)
    pos = rng.uniform(lb, ub, size=(num_particles, dim))

    # Generating initial velocities
    velocities = np.zeros((num_particles, dim))

    # Algorithm
    pbest = pos.copy()
    vbest = np.array([fn_obj(p) for p in pos])

    best_index = np.argmin(vbest)
    gbest = pos[best_index].copy()
    gbest_value = vbest[best_index]

    for i in range(iterations):
        r1 = rng.random((num_particles, dim))
        r2 = rng.random((num_particles, dim))

        # Update velocity
        progress = i / max(1, iterations - 1)
        w = w_start + progress * (w_end - w_start)
        velocities = w * velocities + c1 * r1 * (pbest - pos) + c2 * r2 * (gbest - pos)

        # Update position
        new_pos = pos + velocities
        pos = np.clip(new_pos, lb, ub)

        # Update values
        values = np.array([fn_obj(p) for p in pos])

        # Update pbest
        pbest[values < vbest] = pos[values < vbest]
        vbest[values < vbest] = values[values < vbest]

        # Update gbest
        best_index = np.argmin(vbest)
        if vbest[best_index] < gbest_value:
            gbest = pbest[best_index].copy()
            gbest_value = vbest[best_index]

    return {
        "best position": gbest,
        "best value": gbest_value,
    }


if __name__ == '__main__':
    pass