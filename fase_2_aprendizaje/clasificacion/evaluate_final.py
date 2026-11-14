"""Evaluación final del modelo campeón sobre la partición test.

Protocolo estricto:
1. Reajusta el modelo campeón exclusivamente con train + dev (sin test).
2. Evalúa UNA SOLA VEZ sobre la partición test.
3. Agrega la fila experiment_id="test" en classification_metrics.tsv y classification_predictions.tsv.
4. Genera la matriz de confusión final de test.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import load_npz, vstack
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
RESULTADOS_DIR = Path(__file__).resolve().parent / "resultados"
SEMILLA = 0


def evaluar_en_test():
    exp_path = RESULTADOS_DIR / "experiments.tsv"
    metrics_path = RESULTADOS_DIR / "classification_metrics.tsv"
    preds_path = RESULTADOS_DIR / "classification_predictions.tsv"
    docs_path = DATA_PROCESSED_DIR / "labeled_documents.tsv"

    if not exp_path.exists():
        print("Error: experiments.tsv no existe. Ejecuta primero train_classifiers.py")
        sys.exit(1)

    df_exp = pd.read_csv(exp_path, sep="\t")
    # Buscar el campeón seleccionado con mayor F1
    seleccionados = df_exp[df_exp["selected"] == "yes"].copy()
    if seleccionados.empty:
        seleccionados = df_exp.copy()

    # Extraer el valor numérico exacto de dev_metric "macro-F1=0.xxxx"
    seleccionados["f1_val"] = seleccionados["dev_metric"].str.extract(r"macro-F1=([\d\.]+)").astype(float)
    campeon_row = seleccionados.sort_values(by="f1_val", ascending=False).iloc[0]

    campeon_id = campeon_row["experiment_id"]
    metodo = campeon_row["method"]
    norm = campeon_row["normalization"]
    rep = campeon_row["representation"]
    hparams = campeon_row["hyperparameters"]
    f1_dev = campeon_row["f1_val"]

    print("=" * 80)
    print("EVALUACIÓN FINAL EN TEST DEL MODELO CAMPEÓN")
    print("=" * 80)
    print(f"Modelo Campeón:       {metodo} ({campeon_id})")
    print(f"Normalización:        {norm}")
    print(f"Representación:       {rep}")
    print(f"Hiperparámetros:      {hparams}")
    print(f"Desempeño Dev (F1):   {f1_dev:.4f}")

    # Determinar ruta de matrices
    norm_map = {"tokenizado": "1 Tokenizado", "stop_words": "2 Stop words", "lematizado": "3 Lematizado"}
    # extraer representación y n-grama
    partes_rep = rep.split(" ", 1)
    rep_nombre = partes_rep[0]
    rep_map = {"binary": "Binario", "frecuencia": "Frecuencia", "tfidf": "TD IDF"}
    ngram_nombre = partes_rep[1] if len(partes_rep) > 1 else "unigramas"

    matriz_dir = (
        ROOT_DIR
        / "fase_2_aprendizaje"
        / "matrices"
        / norm_map.get(norm, norm)
        / rep_map.get(rep_nombre, rep_nombre)
        / "20261111T024437810514Z"
        / ngram_nombre
    )

    print(f"Cargando matrices desde: {matriz_dir}")
    X_train = load_npz(matriz_dir / "X_train.npz")
    X_dev = load_npz(matriz_dir / "X_dev.npz")
    X_test = load_npz(matriz_dir / "X_test.npz")

    df_docs = pd.read_csv(docs_path, sep="\t")
    y_train = df_docs.loc[df_docs["split"] == "train", "category"].values
    y_dev = df_docs.loc[df_docs["split"] == "dev", "category"].values
    y_test = df_docs.loc[df_docs["split"] == "test", "category"].values
    test_doc_ids = df_docs.loc[df_docs["split"] == "test", "document_id"].values
    classes = sorted(list(set(y_train)))

    # Unión estricta train + dev
    X_train_dev = vstack([X_train, X_dev])
    y_train_dev = np.concatenate([y_train, y_dev])
    print(f"Matriz reentrenamiento (train + dev): {X_train_dev.shape}")
    print(f"Matriz evaluación final (test):        {X_test.shape}")

    # Reconstrucción del modelo con hiperparámetros óptimos
    if metodo == "LinearSVC":
        # extraer C
        c_val = float(hparams.split("C=")[-1].split(",")[0])
        modelo_final = LinearSVC(C=c_val, max_iter=3000, random_state=SEMILLA)
    elif metodo == "LogisticRegression":
        c_val = float(hparams.split("C=")[-1].split(",")[0])
        modelo_final = LogisticRegression(C=c_val, max_iter=1000, random_state=SEMILLA)
    elif metodo == "MultinomialNB":
        alpha_val = float(hparams.split("alpha=")[-1].split(",")[0])
        modelo_final = MultinomialNB(alpha=alpha_val)
    else:
        raise ValueError(f"Método desconocido: {metodo}")

    print(f"Ajustando {metodo} sobre train + dev...")
    modelo_final.fit(X_train_dev, y_train_dev)

    print("Evaluando predicciones sobre test...")
    y_pred_test = modelo_final.predict(X_test)

    # Métricas finales
    acc_test = accuracy_score(y_test, y_pred_test)
    prec_test = precision_score(y_test, y_pred_test, average="macro", zero_division=0)
    rec_test = recall_score(y_test, y_pred_test, average="macro", zero_division=0)
    f1_test = f1_score(y_test, y_pred_test, average="macro", zero_division=0)

    print("\n" + "=" * 80)
    print("RESULTADOS FINALES EN TEST (TEST SCORES)")
    print("=" * 80)
    print(f"Accuracy:        {acc_test:.4f}")
    print(f"Macro-Precision: {prec_test:.4f}")
    print(f"Macro-Recall:    {rec_test:.4f}")
    print(f"Macro-F1:        {f1_test:.4f}")
    print("=" * 80)

    # Actualizar classification_metrics.tsv
    df_metrics = pd.read_csv(metrics_path, sep="\t")
    # eliminar fila test si ya existiera
    df_metrics = df_metrics[df_metrics["experiment_id"] != "test"]

    test_metric_row = pd.DataFrame([{
        "experiment_id": "test",
        "model": metodo,
        "split": "test",
        "accuracy": round(acc_test, 4),
        "macro_precision": round(prec_test, 4),
        "macro_recall": round(rec_test, 4),
        "macro_f1": round(f1_test, 4),
    }])
    df_metrics = pd.concat([df_metrics, test_metric_row], ignore_index=True)
    df_metrics.to_csv(metrics_path, sep="\t", index=False)
    print(f"[OK] Fila 'test' añadida a: {metrics_path}")

    # Actualizar classification_predictions.tsv
    df_preds = pd.read_csv(preds_path, sep="\t")
    df_preds = df_preds[df_preds["experiment_id"] != "test"]

    test_preds_rows = []
    for doc_id, gold, pred in zip(test_doc_ids, y_test, y_pred_test):
        test_preds_rows.append({
            "experiment_id": "test",
            "model": metodo,
            "document_id": doc_id,
            "gold_label": gold,
            "predicted_label": pred,
            "split": "test",
        })
    df_test_preds = pd.DataFrame(test_preds_rows)
    df_preds = pd.concat([df_preds, df_test_preds], ignore_index=True)
    df_preds.to_csv(preds_path, sep="\t", index=False)
    print(f"[OK] Predicciones de test añadidas a: {preds_path}")

    # Reporte por clase en test
    print("\nReporte por Clase en Test:")
    print(classification_report(y_test, y_pred_test, target_names=classes, digits=4))

    # Matriz de confusión en test (reemplaza bloque previo de test si existiera)
    cm_test = confusion_matrix(y_test, y_pred_test, labels=classes)
    cm_df = pd.DataFrame(cm_test, index=classes, columns=classes)
    cm_path = RESULTADOS_DIR / "confusion_matrices.txt"
    marcador_test = "=== Matriz de Confusión Final en TEST"
    contenido_previo = cm_path.read_text(encoding="utf-8") if cm_path.exists() else ""
    if marcador_test in contenido_previo:
        contenido_previo = contenido_previo.split(marcador_test)[0].rstrip()
    nuevo_contenido = (
        contenido_previo
        + f"\n\n=== Matriz de Confusión Final en TEST - {metodo} (experiment_id=test) ===\n"
        + cm_df.to_string()
        + "\n"
    )
    cm_path.write_text(nuevo_contenido, encoding="utf-8")
    print(f"[OK] Matriz de confusión de test guardada en: {cm_path}")


if __name__ == "__main__":
    evaluar_en_test()

