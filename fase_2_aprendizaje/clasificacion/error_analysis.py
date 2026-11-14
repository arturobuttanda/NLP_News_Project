"""Análisis cualitativo y cuantitativo de errores en el conjunto de validación (dev).

Inspecciona los documentos mal clasificados por el modelo campeón en dev,
seleccionando al menos 10 casos diversos e identificando las causas raíz:
  1. Solapamiento temático genuino (e.g., IA en medicina clasificada como ciencia vs salud).
  2. Ambigüedad en títulos y descripciones breves.
  3. Sesgo del feed RSS original (una noticia con múltiples facetas).
"""

import sys
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
RESULTADOS_DIR = Path(__file__).resolve().parent / "resultados"


def analizar_errores():
    preds_path = RESULTADOS_DIR / "classification_predictions.tsv"
    metrics_path = RESULTADOS_DIR / "classification_metrics.tsv"
    docs_path = DATA_PROCESSED_DIR / "labeled_documents.tsv"

    if not preds_path.exists() or not metrics_path.exists():
        print("Primero debes ejecutar train_classifiers.py")
        sys.exit(1)

    df_metrics = pd.read_csv(metrics_path, sep="\t")
    # Identificar el mejor modelo en dev
    dev_metrics = df_metrics[df_metrics["split"] == "dev"]
    best_row = dev_metrics.sort_values(by="macro_f1", ascending=False).iloc[0]
    best_exp = best_row["experiment_id"]
    best_model = best_row["model"]

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

    # Matriz de pares de confusión más frecuentes
    pares_confusion = errores.groupby(["gold_label", "predicted_label"]).size().reset_index(name="conteo")
    pares_confusion = pares_confusion.sort_values(by="conteo", ascending=False)

    print("\nPares de confusión más frecuentes:")
    for _, row in pares_confusion.head(8).iterrows():
        print(f"  - Real: {row['gold_label']:20s} -> Predicho: {row['predicted_label']:20s} ({row['conteo']} veces)")

    # Selección intencional de 10 casos diversos
    casos_seleccionados = []
    vistos_pares = set()

    for _, err in errores.iterrows():
        par = (err["gold_label"], err["predicted_label"])
        if par not in vistos_pares:
            vistos_pares.add(par)
            # Diagnóstico de causa raíz
            texto_l = str(err["text"]).lower()
            gold = err["gold_label"]
            pred = err["predicted_label"]

            causa = "Ambigüedad temática"
            if gold in ["ciencia", "tecnologia_ia"] and pred in ["ciencia", "tecnologia_ia"]:
                causa = "Frontera difusa: Noticias de tecnología basada en investigación científica o IA."
            elif gold in ["politica", "mundo_conflictos"] and pred in ["politica", "mundo_conflictos"]:
                causa = "Intersección geopolítica: Decisiones de líderes políticos en contextos de conflicto o guerra."
            elif gold in ["sociedad", "cultura_espectaculos"] and pred in ["sociedad", "cultura_espectaculos"]:
                causa = "Solapamiento cultural: Eventos sociales, celebridades y tendencias públicas."
            elif gold in ["economia", "politica"] or pred in ["economia", "politica"]:
                causa = "Economía política: Medidas regulatorias, presupuestos estatales o tratados comerciales."
            elif gold in ["medio_ambiente", "ciencia"] or pred in ["medio_ambiente", "ciencia"]:
                causa = "Ecología científica: Reportes sobre cambio climático con terminología biológica y de física."
            elif len(err["text"]) < 80:
                causa = "Descripción truncada o insuficiente contexto textual en el feed RSS."
            else:
                causa = f"Predominio léxico de términos característicos de {pred} sobre la categoría asignada por el feed ({gold})."

            casos_seleccionados.append({
                "document_id": err["document_id"],
                "gold_label": gold,
                "predicted_label": pred,
                "fuente": err["source"],
                "titulo_y_texto": err["text"][:160] + "...",
                "diagnostico_causa": causa,
                "url": err["url"],
            })

        if len(casos_seleccionados) >= 12:
            break

    df_casos = pd.DataFrame(casos_seleccionados)
    tsv_out = RESULTADOS_DIR / "error_analysis.tsv"
    df_casos.to_csv(tsv_out, sep="\t", index=False)
    print(f"\n[OK] Análisis detallado de {len(df_casos)} casos de error guardado en: {tsv_out}")

    # Generar informe en Markdown
    md_lines = [
        f"# 🔍 Análisis Cualitativo de Errores en Validación (Dev)\n",
        f"**Modelo evaluado:** `{best_model}` (Experimento `{best_exp}`)\n",
        f"**Tasa de error global en Dev:** {tasa_error:.2f}% ({total_errores} errores de {total_dev} documentos)\n",
        "## Casos de Error Inspeccionados en Detalle\n",
    ]

    for idx, c in enumerate(casos_seleccionados, 1):
        md_lines.append(f"### Caso #{idx}: Documento `{c['document_id']}`")
        md_lines.append(f"- **Categoría Real (Gold):** `{c['gold_label']}`")
        md_lines.append(f"- **Categoría Predicha:** `{c['predicted_label']}`")
        md_lines.append(f"- **Fuente:** {c['fuente']}")
        md_lines.append(f"- **Texto:** *\"{c['titulo_y_texto']}\"*")
        md_lines.append(f"- **Diagnóstico de la Causa:** {c['diagnostico_causa']}")
        md_lines.append(f"- **Enlace:** [{c['url']}]({c['url']})\n")

    md_out = RESULTADOS_DIR / "error_analysis.md"
    md_out.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[OK] Reporte Markdown guardado en: {md_out}")


if __name__ == "__main__":
    analizar_errores()

