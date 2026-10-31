import numpy as np

from utils.methods import _halton_sequence, _levy_flight


def WOSolver(
    lb,
    ub,
    fn_obj,
    num_walruses=10,
    T=50,
    dim=2,
    seed=None,
    return_history=False,
):
    if dim < 1:
        raise ValueError("dim must be at least 1")
    if num_walruses < 3:
        raise ValueError("num_walruses must be at least 3")
    if T < 1:
        raise ValueError("iterations must be at least 1")

    lb = np.broadcast_to(np.asarray(lb, dtype=float), (dim,)).copy()
    ub = np.broadcast_to(np.asarray(ub, dtype=float), (dim,)).copy()
    if not np.all(np.isfinite(lb)) or not np.all(np.isfinite(ub)):
        raise ValueError("Bounds must be finite")
    if np.any(lb >= ub):
        raise ValueError("Each lower bound must be smaller than its upper bound")

    rng = np.random.default_rng(seed)
    pos = rng.uniform(lb, ub, size=(num_walruses, dim))
    values = np.array([fn_obj(walrus) for walrus in pos], dtype=float)

    # Keep the requested juvenile share and divide all remaining adults
    # between male and female roles, including odd adult counts.
    num_juveniles = max(1, int(round(0.1 * num_walruses)))
    num_adults = num_walruses - num_juveniles
    num_males = (num_adults + 1) // 2
    num_females = num_adults - num_males

    role_order = rng.permutation(num_walruses)
    male_indices = role_order[:num_males]
    female_indices = role_order[num_males:num_adults]
    juvenile_indices = role_order[num_adults:]

    best_index = np.argmin(values)
    best_pos = pos[best_index].copy()
    best_value = float(values[best_index])

    if return_history:
        position_history = [pos.copy()]
        trajectory = [pos[0].copy()]
        average_fitness = [float(np.mean(values))]
        convergence = [best_value]

    for t in range(T):
        progress = t / T
        alpha = 1.0 - progress
        r1 = rng.uniform(0.0, 1.0)
        r2 = rng.uniform(0.0, 1.0)
        r3 = rng.uniform(0.0, 1.0)
        r4 = rng.uniform(0.0, 1.0)
        r5 = rng.uniform(0.0, 1.0)

        A = 2.0 * alpha
        R = 2.0 * r1 - 1.0
        danger_signal = A * R
        safety_signal = r2
        beta = 1.0 / (1.0 + np.exp(10.0 * (progress - 0.5)))
        new_pos = pos.copy()

        if abs(danger_signal) >= 1.0:
            for i in range(num_walruses):
                Xn, Xm = rng.choice(num_walruses, size=2, replace=False)
                migration_step = beta * (r3 ** 2) * (pos[Xn] - pos[Xm])
                new_pos[i] = pos[i] + migration_step

        elif safety_signal >= 0.5:
            male_positions = lb + _halton_sequence(
                num_males,
                dim,
                start_index=t * num_males + 1,
            ) * (ub - lb)
            new_pos[male_indices] = male_positions

            female_positions = pos[female_indices]
            selected_males = pos[
                rng.choice(male_indices, size=num_females, replace=True)
            ]
            new_pos[female_indices] = (
                female_positions
                + alpha * (selected_males - female_positions)
                + (1.0 - alpha) * (best_pos - female_positions)
            )

            juvenile_positions = pos[juvenile_indices]
            levy = _levy_flight(rng, size=(num_juveniles, dim))
            safety_position = best_pos + juvenile_positions * levy
            distress_factor = rng.uniform(
                0.0,
                1.0,
                size=(num_juveniles, 1),
            )
            new_pos[juvenile_indices] = distress_factor * (
                safety_position - juvenile_positions
            )

        elif abs(danger_signal) >= 0.5:
            new_pos = (
                pos * R
                - np.abs(best_pos - pos) * (r4 ** 2)
            )

        else:
            a = beta * r5 - beta
            theta = rng.uniform(0.0, np.pi, size=(num_walruses, dim))
            second_idx = np.argsort(values)[1]
            X1 = best_pos - a * np.tan(theta) * np.abs(best_pos - pos)
            X2 = (
                pos[second_idx]
                - a
                * np.tan(theta)
                * np.abs(pos[second_idx] - pos)
            )
            new_pos = 0.5 * (X1 + X2)

        pos = np.clip(new_pos, lb, ub)
        values = np.array([fn_obj(walrus) for walrus in pos], dtype=float)

        current_best = np.argmin(values)
        if values[current_best] < best_value:
            best_pos = pos[current_best].copy()
            best_value = float(values[current_best])

        if return_history:
            position_history.append(pos.copy())
            trajectory.append(pos[0].copy())
            average_fitness.append(float(np.mean(values)))
            convergence.append(best_value)

    result = {
        "best position": best_pos,
        "best value": best_value,
    }
    if return_history:
        result.update({
            "history": np.asarray(position_history),
            "trajectory": np.asarray(trajectory),
            "average_fitness": np.asarray(average_fitness),
            "convergence": np.asarray(convergence),
        })
    return result
