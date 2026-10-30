import numpy as np

def _make_qualitative_plots(benchmark, result, output_dir):
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Instala matplotlib para crear las gráficas cualitativas: python -m pip install matplotlib") from error

    output_dir.mkdir(parents=True, exist_ok=True)
    history = np.asarray(result["history"])
    x_label = "Coordenada 1"
    y_label = "Coordenada 2"

    figure, axis = plt.subplots()
    points = history[:, :, :2].reshape(-1, 2)
    iterations = np.repeat(np.arange(history.shape[0]), history.shape[1])
    scatter = axis.scatter(points[:, 0], points[:, 1], c=iterations, cmap="viridis", s=18)
    figure.colorbar(scatter, ax=axis, label="Iteración")
    axis.set(title=f"Historial de búsqueda: {benchmark['name']}", xlabel=x_label, ylabel=y_label)
    figure.savefig(output_dir / "search_history.png", bbox_inches="tight")
    plt.close(figure)

    trajectory = np.asarray(result["trajectory"])
    figure, axis = plt.subplots()
    axis.plot(trajectory[:, 0], trajectory[:, 1], marker=".", markersize=3)
    axis.scatter(trajectory[0, 0], trajectory[0, 1], label="Inicio")
    axis.scatter(trajectory[-1, 0], trajectory[-1, 1], label="Final")
    axis.set(title=f"Trayectoria del primer walrus: {benchmark['name']}", xlabel=x_label, ylabel=y_label)
    axis.legend()
    figure.savefig(output_dir / "first_walrus_trajectory.png", bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots()
    axis.plot(result["average_fitness"])
    axis.set(title=f"Fitness promedio de la población: {benchmark['name']}", xlabel="Iteración", ylabel="Fitness promedio")
    figure.savefig(output_dir / "average_fitness.png", bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots()
    axis.plot(result["convergence"])
    axis.set(title=f"Convergencia de WO: {benchmark['name']}", xlabel="Iteración", ylabel="Mejor fitness")
    figure.savefig(output_dir / "convergence.png", bbox_inches="tight")
    plt.close(figure)