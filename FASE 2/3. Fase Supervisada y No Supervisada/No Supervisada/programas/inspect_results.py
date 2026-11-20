"""Interfaz de consulta para los resultados del agrupamiento no supervisado."""

import argparse
from pathlib import Path

import pandas as pd


TASK_DIR = Path(__file__).resolve().parents[1]
RESULTADOS_DIR = TASK_DIR / "resultados"
EXPERIMENTACION_DIR = RESULTADOS_DIR / "01_experimentacion"
ASIGNACIONES_DIR = RESULTADOS_DIR / "02_asignaciones"
INTERPRETACION_DIR = RESULTADOS_DIR / "03_interpretacion"

EXPERIMENTS_PATH = EXPERIMENTACION_DIR / "clustering_experiments.tsv"
ASSIGNMENTS_PATH = ASIGNACIONES_DIR / "clustering_results.tsv"
INTERPRETATION_PATH = INTERPRETACION_DIR / "cluster_interpretation.tsv"


def load_required_tsv(path):
    """Carga un TSV y explica qué ejecutar si aún no fue generado."""
    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path}. Ejecuta primero train_kmeans.py."
        )
    return pd.read_csv(path, sep="\t")


def show_experiments(args):
    """Filtra y ordena el registro completo por Silhouette, NMI y ARI."""
    df = load_required_tsv(EXPERIMENTS_PATH)

    if args.experiment_id:
        df = df[df["experiment_id"].str.upper() == args.experiment_id.upper()]
    if args.normalization:
        df = df[df["normalization"].str.contains(args.normalization, case=False, na=False)]
    if args.representation:
        df = df[df["representation"].str.contains(args.representation, case=False, na=False)]
    if args.method:
        df = df[df["method"].str.contains(args.method, case=False, na=False)]

    df = df.sort_values(
        by=["silhouette", "nmi", "ari"],
        ascending=False,
    ).head(args.limit)

    if df.empty:
        print("No se encontraron experimentos con esos filtros.")
        return

    print(df.to_string(index=False))


def show_selected(args):
    df = load_required_tsv(EXPERIMENTS_PATH)
    selected = df[df["selected"].str.lower() == "yes"]
    if selected.empty:
        print("No hay ninguna configuración marcada como seleccionada.")
        return

    row = selected.iloc[0]
    print(row.to_string())


def show_cluster(args):
    """Muestra el resumen de uno o todos los clústeres."""
    df = load_required_tsv(INTERPRETATION_PATH)
    if args.cluster_id is not None:
        df = df[df["cluster_id"] == args.cluster_id]

    df = df.sort_values("cluster_id")
    if df.empty:
        print("No se encontró el clúster solicitado.")
        return

    print(df.to_string(index=False))


def show_assignments(args):
    df = load_required_tsv(ASSIGNMENTS_PATH)

    if args.cluster_id is not None:
        df = df[df["cluster_id"] == args.cluster_id]
    if args.split:
        df = df[df["split"].str.lower() == args.split.lower()]

    if df.empty:
        print("No se encontraron asignaciones con esos filtros.")
        return

    print(df.head(args.limit).to_string(index=False))
    print(f"\nAsignaciones mostradas: {min(args.limit, len(df))}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Consulta los resultados del agrupamiento no supervisado."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    experiments = subparsers.add_parser(
        "experiments",
        help="Consulta métricas de los experimentos.",
    )
    experiments.add_argument("--experiment-id")
    experiments.add_argument("--normalization")
    experiments.add_argument("--representation")
    experiments.add_argument("--method")
    experiments.add_argument("--limit", type=int, default=20)
    experiments.set_defaults(handler=show_experiments)

    selected = subparsers.add_parser(
        "selected",
        help="Muestra la configuración seleccionada.",
    )
    selected.set_defaults(handler=show_selected)

    cluster = subparsers.add_parser(
        "cluster",
        help="Muestra tamaño, palabras y documentos representativos.",
    )
    cluster.add_argument("--cluster-id", type=int)
    cluster.set_defaults(handler=show_cluster)

    assignments = subparsers.add_parser(
        "assignments",
        help="Consulta asignaciones de documentos.",
    )
    assignments.add_argument("--cluster-id", type=int)
    assignments.add_argument("--split", choices=("train", "dev", "test"))
    assignments.add_argument("--limit", type=int, default=20)
    assignments.set_defaults(handler=show_assignments)

    return parser


if __name__ == "__main__":
    parser = build_parser()
    arguments = parser.parse_args()
    arguments.handler(arguments)
