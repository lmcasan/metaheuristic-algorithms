from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def make_qualitative_plots(benchmark, result, output_dir):

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
        y_label = "koordinatsız"
    else:
        points = history[:, :, :2].reshape(-1, 2)
        trajectory_points = trajectory[:, :2]
        y_label = "Koordinat 2"

    x_label = "Koordinat 1"
    iterations = np.repeat(np.arange(history.shape[0]), history.shape[1])

    figure, axis = plt.subplots()
    scatter = axis.scatter(
        points[:, 0],
        points[:, 1],
        c=iterations,
        cmap="viridis",
        s=18,
    )
    figure.colorbar(scatter, ax=axis, label="Iterasyon")
    axis.set(
        title=f"Arama Geçmisi: {benchmark['name']}",
        xlabel=x_label,
        ylabel=y_label,
    )
    figure.savefig(output_dir / "search_history.png", bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots()
    axis.plot(trajectory_points[:, 0], trajectory_points[:, 1], marker=".", markersize=3)
    axis.scatter(trajectory_points[0, 0], trajectory_points[0, 1], label="başlangıç")
    axis.scatter(trajectory_points[-1, 0], trajectory_points[-1, 1], label="son")
    axis.set(
        title=f"Popülasyondaki ilk mors bireyinin yörüngesi: {benchmark['name']}",
        xlabel=x_label,
        ylabel=y_label,
    )
    axis.legend()
    figure.savefig(output_dir / "first_walrus_trajectory.png", bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots()
    axis.plot(result["average_fitness"])
    axis.set(
        title=f"Uygunluk değeri: {benchmark['name']}",
        xlabel="Iterasyon",
        ylabel="En iyi uygunluk değeri",
    )
    figure.savefig(output_dir / "average_fitness.png", bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots()
    axis.plot(result["convergence"])
    axis.set(
        title=f"WO yakınsama: {benchmark['name']}",
        xlabel="Iterasyon",
        ylabel="En iyi uygunluk değeri",
    )
    figure.savefig(output_dir / "convergence.png", bbox_inches="tight")
    plt.close(figure)
