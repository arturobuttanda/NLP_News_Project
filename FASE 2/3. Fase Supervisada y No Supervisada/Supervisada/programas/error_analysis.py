"""Selecciona casos mal clasificados por el modelo campeón en el conjunto dev."""

import sys
from pathlib import Path
import pandas as pd

TASK_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = Path(__file__).resolve().parents[3]
PREPROCESSING_RESULTS_DIR = ROOT_DIR / "2. Preprocesamiento (Scrapper, Normalizacion, Vectorial)" / "resultados"
CORPUS_DIR = PREPROCESSING_RESULTS_DIR / "3 Lematizado" / "Frecuencia" / "Corpus KM01"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
RESULTADOS_DIR = TASK_DIR / "resultados"
METRICAS_DIR = RESULTADOS_DIR / "02_metricas_y_predicciones"
ERRORES_DIR = RESULTADOS_DIR / "04_analisis_errores"


def analizar_errores():
    """Selecciona hasta doce errores diversos y registra las métricas del campeón."""
    preds_path = METRICAS_DIR / "classification_predictions.tsv"
    metrics_path = METRICAS_DIR / "classification_metrics.tsv"
    # Se consulta la misma tabla etiquetada usada para construir las matrices.
    docs_path = CORPUS_DIR / "labeled_documents.tsv"
    if not docs_path.exists():
        docs_path = DATA_PROCESSED_DIR / "labeled_documents.tsv"

    if not preds_path.exists() or not metrics_path.exists():
        print("Primero debes ejecutar train_classifiers.py")
        sys.exit(1)

    df_metrics = pd.read_csv(metrics_path, sep="\t")
    # Identificar el mejor modelo en dev por Macro-F1.
    dev_metrics = df_metrics[df_metrics["split"] == "dev"]
    best_row = dev_metrics.sort_values(by="macro_f1", ascending=False).iloc[0]
    best_exp = best_row["experiment_id"]
    best_model = best_row["model"]
    champion_metrics = {
        "accuracy": round(float(best_row["accuracy"]), 4),
        "macro_precision": round(float(best_row["macro_precision"]), 4),
        "macro_recall": round(float(best_row["macro_recall"]), 4),
        "macro_f1": round(float(best_row["macro_f1"]), 4),
    }

    print(f"Analizando errores de dev para el modelo campeón: {best_model} ({best_exp})")

    df_preds = pd.read_csv(preds_path, sep="\t")
    df_best_preds = df_preds[(df_preds["experiment_id"] == best_exp) & (df_preds["split"] == "dev")].copy()

    df_docs = pd.read_csv(docs_path, sep="\t")
    merged = pd.merge(df_best_preds, df_docs[["document_id", "text", "source", "url"]], on="document_id")

    # Filtrar discrepancias
    errores = merged[merged["gold_label"] != merged["predicted_label"]].copy()
    total_dev = len(df_best_preds)
    total_errores = len(errores)
    tasa_error = (total_errores / total_dev) * 100

    print(f"Total documentos en dev: {total_dev}")
    print(f"Documentos con error: {total_errores} ({tasa_error:.2f}%)")

    # Selección intencional de hasta doce casos diversos.
    casos_seleccionados = []
    vistos_pares = set()

    for _, err in errores.iterrows():
        par = (err["gold_label"], err["predicted_label"])
        if par not in vistos_pares:
            vistos_pares.add(par)
            casos_seleccionados.append({
                "document_id": err["document_id"],
                "gold_label": err["gold_label"],
                "predicted_label": err["predicted_label"],
                "fuente": err["source"],
                "titulo_y_texto": err["text"][:160] + "...",
                "url": err["url"],
                "model": best_model,
                "experiment_id": best_exp,
                "accuracy": champion_metrics["accuracy"],
                "macro_precision": champion_metrics["macro_precision"],
                "macro_recall": champion_metrics["macro_recall"],
                "macro_f1": champion_metrics["macro_f1"],
                "dev_documents": total_dev,
                "dev_errors": total_errores,
                "dev_error_rate": round(tasa_error, 4),
            })

        if len(casos_seleccionados) >= 12:
            break

    df_casos = pd.DataFrame(casos_seleccionados)
    tsv_out = ERRORES_DIR / "error_analysis.tsv"
    df_casos.to_csv(tsv_out, sep="\t", index=False)
    print(f"\n[OK] Selección de {len(df_casos)} casos de error guardada en: {tsv_out}")


if __name__ == "__main__":
    analizar_errores()

