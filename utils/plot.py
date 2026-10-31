from pathlib import Path

import numpy as np


def make_qualitative_plots(benchmark, result, output_dir):
    import matplotlib.pyplot as plt

    required = ("history", "trajectory", "average_fitness", "convergence")
    missing = [key for key in required if key not in result]
    if missing:
        raise ValueError(
            "WO result is missing history fields required for plotting: "
            + ", ".join(missing)
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history = np.asarray(result["history"])
    trajectory = np.asarray(result["trajectory"])
    dimension = history.shape[2]

    if dimension == 1:
        points = np.column_stack((
            history[:, :, 0].reshape(-1),
            np.zeros(history.shape[0] * history.shape[1]),
        ))
        trajectory_points = np.column_stack((
            trajectory[:, 0],
            np.zeros(trajectory.shape[0]),
        ))
        y_label = "Sin coordenada"
    else:
        points = history[:, :, :2].reshape(-1, 2)
        trajectory_points = trajectory[:, :2]
        y_label = "Coordenada 2"

    x_label = "Coordenada 1"
    iterations = np.repeat(np.arange(history.shape[0]), history.shape[1])

    figure, axis = plt.subplots()
    scatter = axis.scatter(
        points[:, 0],
        points[:, 1],
        c=iterations,
        cmap="viridis",
        s=18,
    )
    figure.colorbar(scatter, ax=axis, label="Iteración")
    axis.set(
        title=f"Historial de búsqueda: {benchmark['name']}",
        xlabel=x_label,
        ylabel=y_label,
    )
    figure.savefig(output_dir / "search_history.png", bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots()
    axis.plot(trajectory_points[:, 0], trajectory_points[:, 1], marker=".", markersize=3)
    axis.scatter(trajectory_points[0, 0], trajectory_points[0, 1], label="Inicio")
    axis.scatter(trajectory_points[-1, 0], trajectory_points[-1, 1], label="Final")
    axis.set(
        title=f"Trayectoria del primer walrus: {benchmark['name']}",
        xlabel=x_label,
        ylabel=y_label,
    )
    axis.legend()
    figure.savefig(output_dir / "first_walrus_trajectory.png", bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots()
    axis.plot(result["average_fitness"])
    axis.set(
        title=f"Fitness promedio de la población: {benchmark['name']}",
        xlabel="Iteración",
        ylabel="Fitness promedio",
    )
    figure.savefig(output_dir / "average_fitness.png", bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots()
    axis.plot(result["convergence"])
    axis.set(
        title=f"Convergencia de WO: {benchmark['name']}",
        xlabel="Iteración",
        ylabel="Mejor fitness",
    )
    figure.savefig(output_dir / "convergence.png", bbox_inches="tight")
    plt.close(figure)
