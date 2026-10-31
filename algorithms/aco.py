import numpy as np


def ACOSolver(
    lb,
    ub,
    fn_obj,
    num_ants=10,
    archive_size=10,
    iterations=50,
    q=0.5,
    xi=0.85,
    dim=2,
    seed=None,
):
    if dim < 1:
        raise ValueError("dim must be at least 1")
    if num_ants < 1:
        raise ValueError("num_ants must be at least 1")
    if archive_size < 2:
        raise ValueError("archive_size must be at least 2")
    if iterations < 1:
        raise ValueError("iterations must be at least 1")
    if q <= 0.0:
        raise ValueError("q must be greater than 0")
    if xi <= 0.0:
        raise ValueError("xi must be greater than 0")

    lb = np.broadcast_to(np.asarray(lb, dtype=float), (dim,)).copy()
    ub = np.broadcast_to(np.asarray(ub, dtype=float), (dim,)).copy()
    if not np.all(np.isfinite(lb)) or not np.all(np.isfinite(ub)):
        raise ValueError("Bounds must be finite")
    if np.any(lb >= ub):
        raise ValueError("Each lower bound must be smaller than its upper bound")

    rng = np.random.default_rng(seed)
    archive = rng.uniform(lb, ub, size=(archive_size, dim))
    values = np.array([fn_obj(ant) for ant in archive], dtype=float)

    ranks = np.arange(archive_size)
    weights = np.exp(-(ranks ** 2) / (2.0 * (q * archive_size) ** 2))
    probabilities = weights / np.sum(weights)

    for _ in range(iterations):
        selected = rng.choice(
            archive_size,
            size=(num_ants, dim),
            p=probabilities,
        )
        coordinates = np.arange(dim)[None, :]
        centers = archive[selected, coordinates]
        spread = np.sum(
            np.abs(archive[None, :, :] - centers[:, None, :]),
            axis=1,
        )
        sigma = xi * spread / (archive_size - 1)
        sigma = np.maximum(sigma, 1e-12 * (ub - lb))
        new_ants = rng.normal(centers, sigma)
        new_ants = np.clip(new_ants, lb, ub)
        new_values = np.array([fn_obj(ant) for ant in new_ants], dtype=float)

        archive = np.vstack((archive, new_ants))
        values = np.concatenate((values, new_values))
        best_indices = np.argsort(values)[:archive_size]
        archive = archive[best_indices].copy()
        values = values[best_indices].copy()

    best_index = np.argmin(values)
    return {
        "best position": archive[best_index].copy(),
        "best value": float(values[best_index]),
    }
