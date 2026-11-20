"""Entrenamiento y optimización de clasificadores supervisados para categorización de noticias.

Implementa de forma reproducible todos los requisitos de la Tarea A (Clasificación Supervisada):
  1. Auditoría de etiquetas (mapeo desde feeds RSS, distribución por clase y control de faltantes/ambiguas).
  2. Optimización con Pipelines de scikit-learn + GridSearchCV (StratifiedKFold en train, scoring='f1_macro').
  3. Barrido exhaustivo controlado de 398 configuraciones sobre las matrices precalculadas (Tokenizado,
     Stop words, Lematizado × Binario, Frecuencia, TF-IDF × N-gramas + TruncatedSVD).
  4. Selección por Macro-F1 en validación (dev), reportes por clase, classification_report y matrices de confusión.

Genera los entregables oficiales en resultados/:
  - grid_search_summary.tsv
  - experiments.tsv
  - classification_metrics.tsv
  - classification_predictions.tsv
  - per_class_metrics.tsv
  - confusion_matrices.txt
"""

import os
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

TASK_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = Path(__file__).resolve().parents[3]
PREPROCESSING_RESULTS_DIR = ROOT_DIR / "2. Preprocesamiento (Scrapper, Normalizacion, Vectorial)" / "resultados"
# Ruta de compatibilidad para la representación lematizada vigente.
CORPUS_DIR = (
    PREPROCESSING_RESULTS_DIR
    / "1. Combinado"
    / "1.4 Stop words Lematizado"
    / "Frecuencia"
    / "unigramas"
)
# Se conservan estas constantes para aceptar también la estructura histórica.
MATRICES_DIR = ROOT_DIR / "fase_2_aprendizaje" / "matrices"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
RESULTADOS_DIR = TASK_DIR / "resultados"
EXPERIMENTACION_DIR = RESULTADOS_DIR / "01_experimentacion"
METRICAS_DIR = RESULTADOS_DIR / "02_metricas_y_predicciones"
REPORTES_DIR = RESULTADOS_DIR / "03_reportes"
ERRORES_DIR = RESULTADOS_DIR / "04_analisis_errores"
for carpeta in [EXPERIMENTACION_DIR, METRICAS_DIR, REPORTES_DIR, ERRORES_DIR]:
    # Se asegura que cada grupo de entregables tenga su carpeta.
    carpeta.mkdir(parents=True, exist_ok=True)

CORRIDA_ID = "20261111T024437810514Z"
SEMILLA = 0

CATEGORIAS_CANONICAS = {
    "ciencia",
    "cultura_espectaculos",
    "deportes",
    "economia",
    "medio_ambiente",
    "mundo_conflictos",
    "politica",
    "salud",
    "sociedad",
    "tecnologia_ia",
}

NORMALIZACIONES = [
    # Esta lista define las líneas de preprocesamiento comparables.
    # La primera variante conserva solo la tokenización y funciona como línea base.
    ("1. Combinado/1. Ind Tokenizado", "tokenizado_minimo"),
    ("1. Combinado/1.1 Comb Stop words", "combinado_stop_words"),
    ("1. Combinado/1.4 Stop words Lematizado", "stopwords_lemas"),
]

REPRESENTACIONES = [
    ("Binario", "binary"),
    ("Frecuencia", "frecuencia"),
    ("TD IDF", "tfidf"),
]

NGRAMAS_PRIORITARIOS = [
    "unigramas",
    "unigramas_bigramas",
    "bigramas",
    "unigramas_bigramas_trigramas",
]


def cargar_y_auditar_etiquetas():
    """Carga el corpus etiquetado y audita mapeo, faltantes, ambigüedades y distribución de clases.

    Procedimiento de mapeo y tratamiento de ambigüedades:
      - Las etiquetas provienen de la taxonomía cerrada de 10 categorías definida en el diccionario
        FEEDS de fase_1_recuperacion/scraper/scraper.py (asignación distante por feed RSS).
      - Si una misma noticia aparece en múltiples feeds (ambigüedad de fuente), scraper.py la
        deduplica por url_hash y title_key asignándole determinísticamente la primera categoría del orden FEEDS.
      - Aquí se verifica que no existan valores nulos, vacíos, etiquetas fuera de catálogo ni IDs duplicados.
    """
    # El corpus etiquetado producido por Preprocesamiento es la entrada oficial.
    labeled_path = (
        PREPROCESSING_RESULTS_DIR
        / "1. Combinado"
        / "1. Ind Tokenizado"
        / "N-gramas"
        / "labeled_documents.tsv"
    )
    # Compatibilidad con ejecuciones antiguas que usaban data/processed.
    if not labeled_path.exists():
        labeled_path = DATA_PROCESSED_DIR / "labeled_documents.tsv"
    if not labeled_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de datos: {labeled_path}")

    df = pd.read_csv(labeled_path, sep="\t", dtype=str, keep_default_na=False)
    total_inicial = len(df)

    # Detección y tratamiento de faltantes / vacíos / etiquetas fuera de catálogo
    mask_texto_vacio = df["text"].str.strip().eq("")
    mask_cat_vacia = df["category"].str.strip().eq("")
    mask_cat_invalida = ~df["category"].isin(CATEGORIAS_CANONICAS)
    mask_id_duplicado = df["document_id"].duplicated(keep="first")

    faltantes = int((mask_texto_vacio | mask_cat_vacia).sum())
    fuera_catalogo = int(mask_cat_invalida.sum())
    duplicados = int(mask_id_duplicado.sum())

    df = df.loc[~(mask_texto_vacio | mask_cat_vacia | mask_cat_invalida | mask_id_duplicado)].copy()

    print("\n--- AUDITORÍA DE ETIQUETAS Y DISTRIBUCIÓN DE CLASES ---")
    print(f"Documentos cargados: {total_inicial} | Conservados tras limpieza: {len(df)}")
    print(f"Faltantes/vacíos eliminados: {faltantes} | Etiquetas ambiguas/inválidas: {fuera_catalogo} | IDs duplicados: {duplicados}")

    dist_splits = pd.crosstab(df["category"], df["split"], margins=True)[["train", "dev", "test", "All"]]
    print("\nDistribución estratificada por categoría y partición (70% train / 15% dev / 15% test):")
    print(dist_splits.to_string())

    conteo_total = df["category"].value_counts()
    ratio_desbalance = conteo_total.max() / conteo_total.min()
    print(f"\nRatio de desbalance (clase mayoritaria / minoritaria): {ratio_desbalance:.2f}x -> justifica Macro-F1.")
    return df


def ajustar_pipelines_con_gridsearch(df_docs):
    """Ejecuta GridSearchCV con Pipelines de scikit-learn exclusivamente sobre train y compara en dev.

    Cada Pipeline encapsula la vectorización (TF-IDF) y el clasificador dentro de StratifiedKFold(5)
    sobre la partición train para evitar cualquier fuga de datos (data leakage) hacia dev o test.
    """
    print("\n" + "=" * 80)
    print("BÚSQUEDA DE HIPERPARÁMETROS CON PIPELINES Y GridSearchCV (SOLO TRAIN -> DEV)")
    print("=" * 80)

    # Las máscaras separan los datos sin permitir que dev participe en el ajuste.
    # X_train_text solo alimenta el CV interno; X_dev_text se reserva para evaluación.
    train_mask = df_docs["split"] == "train"
    dev_mask = df_docs["split"] == "dev"
    X_train_text = df_docs.loc[train_mask, "text"].values
    y_train = df_docs.loc[train_mask, "category"].values
    X_dev_text = df_docs.loc[dev_mask, "text"].values
    y_dev = df_docs.loc[dev_mask, "category"].values

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)

    with TemporaryDirectory() as cache_dir:
        # Cada candidato contiene el vectorizador, el clasificador y su rejilla.
        # El pipeline garantiza que TF-IDF o SVD se ajusten dentro de cada pliegue.
        candidatos = [
            (
                "MultinomialNB",
                Pipeline(
                    [("vect", TfidfVectorizer()), ("clf", MultinomialNB())],
                    memory=cache_dir,
                ),
                {
                    "vect__ngram_range": [(1, 1), (1, 2)],
                    "vect__stop_words": [None, "english"],
                    "clf__alpha": [0.01, 0.1, 0.5, 1.0],
                },
            ),
            (
                "LogisticRegression",
                Pipeline(
                    [("vect", TfidfVectorizer()), ("clf", LogisticRegression(max_iter=500, tol=1e-3, random_state=SEMILLA))],
                    memory=cache_dir,
                ),
                {
                    "vect__ngram_range": [(1, 1), (1, 2)],
                    "vect__stop_words": [None, "english"],
                    "clf__C": [1.0, 5.0, 10.0],
                },
            ),
            (
                "LinearSVC",
                Pipeline(
                    [("vect", TfidfVectorizer()), ("clf", LinearSVC(max_iter=3000, random_state=SEMILLA))],
                    memory=cache_dir,
                ),
                {
                    "vect__ngram_range": [(1, 1), (1, 2)],
                    "vect__stop_words": [None, "english"],
                    "clf__C": [0.1, 0.5, 1.0, 2.0],
                },
            ),
            (
                "RandomForestClassifier+SVD",
                Pipeline(
                    [
                        ("vect", TfidfVectorizer(ngram_range=(1, 2), stop_words="english")),
                        ("svd", TruncatedSVD(n_components=100, random_state=SEMILLA)),
                        ("clf", RandomForestClassifier(random_state=SEMILLA, n_jobs=1)),
                    ],
                    memory=cache_dir,
                ),
                {
                    "clf__n_estimators": [50, 100],
                    "clf__max_depth": [15, 25],
                },
            ),
        ]

        resumen_gs = []
        # Se ajusta y compara cada familia de clasificador.
        for nombre, pipeline, param_grid in candidatos:
            t0 = time.time()
            grid = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                cv=cv,
                scoring="f1_macro",
                n_jobs=-1,
                refit=True,
            )
            grid.fit(X_train_text, y_train)
            tiempo = time.time() - t0

            preds_dev = grid.best_estimator_.predict(X_dev_text)
            dev_acc = accuracy_score(y_dev, preds_dev)
            dev_f1 = f1_score(y_dev, preds_dev, average="macro", zero_division=0)

            print(f"\n[{nombre}] Tiempo GridSearchCV: {tiempo:.2f}s")
            print(f"  Mejores hiperparámetros (CV Train): {grid.best_params_}")
            print(f"  CV Macro-F1 (5-fold Train): {grid.best_score_:.4f} | Dev Accuracy: {dev_acc:.4f} | Dev Macro-F1: {dev_f1:.4f}")

            resumen_gs.append({
                "model": nombre,
                "best_params": str(grid.best_params_),
                "cv_train_macro_f1": round(grid.best_score_, 4),
                "dev_accuracy": round(dev_acc, 4),
                "dev_macro_f1": round(dev_f1, 4),
                "time_s": round(tiempo, 2),
            })

    gs_df = pd.DataFrame(resumen_gs).sort_values(by="dev_macro_f1", ascending=False)
    gs_path = EXPERIMENTACION_DIR / "grid_search_summary.tsv"
    gs_df.to_csv(gs_path, sep="\t", index=False)
    print(f"\n[OK] Resumen de GridSearchCV guardado en: {gs_path}")
    return gs_df


def evaluar_modelo(modelo, X_train, y_train, X_dev, y_dev):
    """Entrena un modelo y calcula sus métricas sobre la partición dev."""
    # Se mide únicamente el tiempo de ajuste del clasificador.
    # El vocabulario/vectorización ya se ajustó antes al construir las matrices.
    t0 = time.time()
    modelo.fit(X_train, y_train)
    tiempo_entrenamiento = time.time() - t0

    preds_dev = modelo.predict(X_dev)
    acc = accuracy_score(y_dev, preds_dev)
    prec_macro = precision_score(y_dev, preds_dev, average="macro", zero_division=0)
    rec_macro = recall_score(y_dev, preds_dev, average="macro", zero_division=0)
    f1_macro = f1_score(y_dev, preds_dev, average="macro", zero_division=0)

    return {
        "accuracy": acc,
        "macro_precision": prec_macro,
        "macro_recall": rec_macro,
        "macro_f1": f1_macro,
        "tiempo": tiempo_entrenamiento,
        "predicciones": preds_dev,
    }


def ejecutar_experimentos():
    """Ejecuta el barrido de configuraciones y guarda todos los reportes supervisados."""
    print("=" * 80)
    print("INICIANDO ENTRENAMIENTO Y OPTIMIZACIÓN SUPERVISADA (FASE 2)")
    print("=" * 80)

    df_docs = cargar_y_auditar_etiquetas()
    ajustar_pipelines_con_gridsearch(df_docs)

    print("\n" + "=" * 80)
    print("BARRIDO EXHAUSTIVO SOBRE LAS MATRICES DE PREPROCESAMIENTO")
    print("=" * 80)
    train_mask = df_docs["split"] == "train"
    dev_mask = df_docs["split"] == "dev"

    y_train = df_docs.loc[train_mask, "category"].values
    y_dev = df_docs.loc[dev_mask, "category"].values
    dev_doc_ids = df_docs.loc[dev_mask, "document_id"].values

    classes = sorted(list(set(y_train)))
    print(f"Total documentos: {len(df_docs)} (Train: {len(y_train)}, Dev: {len(y_dev)})")
    print(f"Categorías ({len(classes)}): {classes}")

    registro_experimentos = []
    mejores_por_algoritmo = {
        "MultinomialNB": {"f1": -1.0, "exp": None},
        "LogisticRegression": {"f1": -1.0, "exp": None},
        "LinearSVC": {"f1": -1.0, "exp": None},
    }

    exp_counter = 1

    # Barrido sistemático de espacios vectoriales disponibles.
    # El conjunto evita evaluar dos veces la misma ruta de matriz.
    matrices_visitadas = set()
    # Primer factor: variantes de preprocesamiento.
    for norm_dir, norm_label in NORMALIZACIONES:
        # Segundo factor: binario, frecuencia y TF-IDF.
        for rep_dir, rep_label in REPRESENTACIONES:
            # Tercer factor: combinaciones de unigramas, bigramas y trigramas.
            for ngram in NGRAMAS_PRIORITARIOS:
                matriz_path = PREPROCESSING_RESULTS_DIR / norm_dir / rep_dir / ngram
                if not matriz_path.exists() or not (matriz_path / "X_train.npz").exists():
                    continue
                # La instantánea actual representa una sola configuración; no se duplica.
                if matriz_path in matrices_visitadas:
                    continue
                matrices_visitadas.add(matriz_path)

                X_train = load_npz(matriz_path / "X_train.npz")
                X_dev = load_npz(matriz_path / "X_dev.npz")

                # 1. MultinomialNB (requiere no-negativos; funciona con Frecuencia, TF-IDF y Binario)
                # Cada alpha controla el suavizado del clasificador NB.
                for alpha in [0.01, 0.1, 0.5, 1.0]:
                    exp_id = f"C{exp_counter:02d}"
                    exp_counter += 1
                    modelo = MultinomialNB(alpha=alpha)
                    res = evaluar_modelo(modelo, X_train, y_train, X_dev, y_dev)

                    reg = {
                        "experiment_id": exp_id,
                        "task": "classification",
                        "normalization": norm_label,
                        "representation": f"{rep_label} {ngram}",
                        "method": "MultinomialNB",
                        "hyperparameters": f"alpha={alpha}",
                        "dev_metric": f"macro-F1={res['macro_f1']:.4f}",
                        "macro_f1": res["macro_f1"],
                        "accuracy": res["accuracy"],
                        "macro_precision": res["macro_precision"],
                        "macro_recall": res["macro_recall"],
                        "predicciones": res["predicciones"],
                        "selected": "no",
                        "matriz_path": str(matriz_path),
                        "modelo_obj": modelo,
                    }
                    registro_experimentos.append(reg)

                    if res["macro_f1"] > mejores_por_algoritmo["MultinomialNB"]["f1"]:
                        mejores_por_algoritmo["MultinomialNB"]["f1"] = res["macro_f1"]
                        mejores_por_algoritmo["MultinomialNB"]["exp"] = reg

                # 2. LinearSVC
                # Cada C controla la regularización de LinearSVC.
                for C in [0.1, 0.5, 1.0, 2.0]:
                    exp_id = f"C{exp_counter:02d}"
                    exp_counter += 1
                    modelo = LinearSVC(C=C, max_iter=2000, random_state=SEMILLA)
                    res = evaluar_modelo(modelo, X_train, y_train, X_dev, y_dev)

                    reg = {
                        "experiment_id": exp_id,
                        "task": "classification",
                        "normalization": norm_label,
                        "representation": f"{rep_label} {ngram}",
                        "method": "LinearSVC",
                        "hyperparameters": f"C={C}",
                        "dev_metric": f"macro-F1={res['macro_f1']:.4f}",
                        "macro_f1": res["macro_f1"],
                        "accuracy": res["accuracy"],
                        "macro_precision": res["macro_precision"],
                        "macro_recall": res["macro_recall"],
                        "predicciones": res["predicciones"],
                        "selected": "no",
                        "matriz_path": str(matriz_path),
                        "modelo_obj": modelo,
                    }
                    registro_experimentos.append(reg)

                    if res["macro_f1"] > mejores_por_algoritmo["LinearSVC"]["f1"]:
                        mejores_por_algoritmo["LinearSVC"]["f1"] = res["macro_f1"]
                        mejores_por_algoritmo["LinearSVC"]["exp"] = reg

                # 3. LogisticRegression (exploración en configuraciones representativas)
                # Cada C controla la regularización de regresión logística.
                for C in [1.0, 5.0, 10.0]:
                    exp_id = f"C{exp_counter:02d}"
                    exp_counter += 1
                    modelo = LogisticRegression(C=C, max_iter=500, random_state=SEMILLA)
                    res = evaluar_modelo(modelo, X_train, y_train, X_dev, y_dev)

                    reg = {
                        "experiment_id": exp_id,
                        "task": "classification",
                        "normalization": norm_label,
                        "representation": f"{rep_label} {ngram}",
                        "method": "LogisticRegression",
                        "hyperparameters": f"C={C}",
                        "dev_metric": f"macro-F1={res['macro_f1']:.4f}",
                        "macro_f1": res["macro_f1"],
                        "accuracy": res["accuracy"],
                        "macro_precision": res["macro_precision"],
                        "macro_recall": res["macro_recall"],
                        "predicciones": res["predicciones"],
                        "selected": "no",
                        "matriz_path": str(matriz_path),
                        "modelo_obj": modelo,
                    }
                    registro_experimentos.append(reg)

                    if res["macro_f1"] > mejores_por_algoritmo["LogisticRegression"]["f1"]:
                        mejores_por_algoritmo["LogisticRegression"]["f1"] = res["macro_f1"]
                        mejores_por_algoritmo["LogisticRegression"]["exp"] = reg

    # 4. Experimento con Reducción de Dimensionalidad (TruncatedSVD)
    print("\nEvaluando variante con reducción de dimensionalidad (TruncatedSVD)...")
    # Se parte de la mejor matriz de LinearSVC para aislar el efecto de SVD.
    mejores_matriz = Path(mejores_por_algoritmo["LinearSVC"]["exp"]["matriz_path"])
    X_train_opt = load_npz(mejores_matriz / "X_train.npz")
    X_dev_opt = load_npz(mejores_matriz / "X_dev.npz")

    # Se compara el efecto de dos tamaños de proyección SVD.
    # n=100 y n=300 permiten comparar una reducción alta y una moderada.
    for n_comp in [100, 300]:
        exp_id = f"C{exp_counter:02d}"
        exp_counter += 1
        svd = TruncatedSVD(n_components=n_comp, random_state=SEMILLA)
        X_tr_svd = svd.fit_transform(X_train_opt)
        X_dv_svd = svd.transform(X_dev_opt)

        clf_svd = LinearSVC(C=1.0, max_iter=2000, random_state=SEMILLA)
        res_svd = evaluar_modelo(clf_svd, X_tr_svd, y_train, X_dv_svd, y_dev)

        reg = {
            "experiment_id": exp_id,
            "task": "classification",
            "normalization": mejores_por_algoritmo["LinearSVC"]["exp"]["normalization"],
            "representation": f"{mejores_por_algoritmo['LinearSVC']['exp']['representation']} + SVD({n_comp})",
            "method": "LinearSVC+SVD",
            "hyperparameters": f"n_components={n_comp}, C=1.0",
            "dev_metric": f"macro-F1={res_svd['macro_f1']:.4f}",
            "macro_f1": res_svd["macro_f1"],
            "accuracy": res_svd["accuracy"],
            "macro_precision": res_svd["macro_precision"],
            "macro_recall": res_svd["macro_recall"],
            "predicciones": res_svd["predicciones"],
            "selected": "no",
            "matriz_path": str(mejores_matriz),
            "modelo_obj": clf_svd,
        }
        registro_experimentos.append(reg)

    # Marcar los seleccionados
    # Solo se marcan como seleccionados los mejores de cada algoritmo.
    # Macro-F1 decidido sobre dev es el único criterio de mejor por familia.
    for algo, info in mejores_por_algoritmo.items():
        info["exp"]["selected"] = "yes"

    print("\n" + "=" * 80)
    print("RESUMEN DE MEJORES CONFIGURACIONES EN VALIDACIÓN (DEV)")
    print("=" * 80)
    for algo, info in mejores_por_algoritmo.items():
        e = info["exp"]
        print(f"[{algo}] Exp: {e['experiment_id']} | Macro-F1: {e['macro_f1']:.4f} | Acc: {e['accuracy']:.4f}")
        print(f"       Norm: {e['normalization']} | Rep: {e['representation']} | Params: {e['hyperparameters']}")

    campeon_algo = max(mejores_por_algoritmo.keys(), key=lambda k: mejores_por_algoritmo[k]["f1"])
    campeon_exp = mejores_por_algoritmo[campeon_algo]["exp"]
    print(f"\nMODELO CAMPEÓN GLOBAL: {campeon_algo} ({campeon_exp['experiment_id']}) con Macro-F1 = {campeon_exp['macro_f1']:.4f}")

    baseline_candidates = [
        reg for reg in registro_experimentos
        if reg["normalization"] == "tokenizado_minimo"
        and reg["method"] != "LinearSVC+SVD"
    ]
    if baseline_candidates:
        baseline_exp = max(baseline_candidates, key=lambda reg: reg["macro_f1"])
        baseline_comparison = pd.DataFrame([{
            "baseline_role": "only_tokenization",
            "baseline_experiment_id": baseline_exp["experiment_id"],
            "baseline_method": baseline_exp["method"],
            "baseline_representation": baseline_exp["representation"],
            "baseline_hyperparameters": baseline_exp["hyperparameters"],
            "baseline_macro_f1": round(baseline_exp["macro_f1"], 4),
            "champion_experiment_id": campeon_exp["experiment_id"],
            "champion_method": campeon_exp["method"],
            "champion_representation": campeon_exp["representation"],
            "champion_hyperparameters": campeon_exp["hyperparameters"],
            "champion_macro_f1": round(campeon_exp["macro_f1"], 4),
            "macro_f1_delta": round(campeon_exp["macro_f1"] - baseline_exp["macro_f1"], 4),
        }])
        baseline_path = EXPERIMENTACION_DIR / "baseline_comparison.tsv"
        baseline_comparison.to_csv(baseline_path, sep="\t", index=False)
        print(f"[OK] Comparación con tokenización mínima guardada en: {baseline_path}")

    exp_df = pd.DataFrame(registro_experimentos)[
        ["experiment_id", "task", "normalization", "representation", "method", "hyperparameters", "dev_metric", "selected"]
    ]
    exp_tsv_path = EXPERIMENTACION_DIR / "experiments.tsv"
    exp_df.to_csv(exp_tsv_path, sep="\t", index=False)
    print(f"\n[OK] Registro de {len(exp_df)} experimentos guardado en: {exp_tsv_path}")

    # Guardar classification_metrics.tsv para los mejores de cada algoritmo.
    metrics_rows = []
    for algo in ["MultinomialNB", "LogisticRegression", "LinearSVC"]:
        e = mejores_por_algoritmo[algo]["exp"]
        metrics_rows.append({
            "experiment_id": e["experiment_id"],
            "model": algo,
            "split": "dev",
            "accuracy": round(e["accuracy"], 4),
            "macro_precision": round(e["macro_precision"], 4),
            "macro_recall": round(e["macro_recall"], 4),
            "macro_f1": round(e["macro_f1"], 4),
        })

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_tsv_path = METRICAS_DIR / "classification_metrics.tsv"
    metrics_df.to_csv(metrics_tsv_path, sep="\t", index=False)
    print(f"[OK] Métricas de modelos seleccionados guardadas en: {metrics_tsv_path}")

    # Guardar classification_predictions.tsv para los mejores de cada algoritmo.
    preds_rows = []
    for algo in ["MultinomialNB", "LogisticRegression", "LinearSVC"]:
        e = mejores_por_algoritmo[algo]["exp"]
        for doc_id, gold, pred in zip(dev_doc_ids, y_dev, e["predicciones"]):
            preds_rows.append({
                "experiment_id": e["experiment_id"],
                "model": algo,
                "document_id": doc_id,
                "gold_label": gold,
                "predicted_label": pred,
                "split": "dev",
            })

    preds_df = pd.DataFrame(preds_rows)
    preds_tsv_path = METRICAS_DIR / "classification_predictions.tsv"
    preds_df.to_csv(preds_tsv_path, sep="\t", index=False)
    print(f"[OK] Predicciones de dev guardadas en: {preds_tsv_path}")

    # Generar reportes por clase (classification_report completo) y matrices de confusión
    per_class_list = []
    cm_text_blocks = []
    report_text_blocks = []

    print("\n" + "=" * 80)
    print("CLASSIFICATION REPORTS EN DEV PARA MODELOS SELECCIONADOS")
    print("=" * 80)

    for algo in ["MultinomialNB", "LogisticRegression", "LinearSVC"]:
        e = mejores_por_algoritmo[algo]["exp"]
        report_str = classification_report(
            y_dev, e["predicciones"], target_names=classes, digits=4, zero_division=0
        )
        print(f"\n--- {algo} ({e['experiment_id']}) | Accuracy: {e['accuracy']:.4f} | Macro-F1: {e['macro_f1']:.4f} ---")
        print(report_str)
        report_text_blocks.append(
            f"=== Classification Report (DEV) - {algo} ({e['experiment_id']}) ===\n"
            f"Norm: {e['normalization']} | Rep: {e['representation']} | Params: {e['hyperparameters']}\n"
            f"Accuracy: {e['accuracy']:.4f} | Macro-Precision: {e['macro_precision']:.4f} | "
            f"Macro-Recall: {e['macro_recall']:.4f} | Macro-F1: {e['macro_f1']:.4f}\n\n"
            + report_str
            + "\n\n"
        )

        report_dict = classification_report(y_dev, e["predicciones"], output_dict=True, zero_division=0)
        for cat in classes:
            if cat in report_dict:
                per_class_list.append({
                    "experiment_id": e["experiment_id"],
                    "model": algo,
                    "category": cat,
                    "precision": round(report_dict[cat]["precision"], 4),
                    "recall": round(report_dict[cat]["recall"], 4),
                    "f1_score": round(report_dict[cat]["f1-score"], 4),
                    "support": int(report_dict[cat]["support"]),
                })

        cm = confusion_matrix(y_dev, e["predicciones"], labels=classes)
        cm_df = pd.DataFrame(cm, index=classes, columns=classes)
        cm_text_blocks.append(f"=== Matriz de Confusión - {algo} ({e['experiment_id']}) ===\n" + cm_df.to_string() + "\n\n")

    per_class_df = pd.DataFrame(per_class_list)
    per_class_path = METRICAS_DIR / "per_class_metrics.tsv"
    per_class_df.to_csv(per_class_path, sep="\t", index=False)
    print(f"[OK] Métricas detalladas por clase guardadas en: {per_class_path}")

    reports_path = REPORTES_DIR / "dev_classification_reports.txt"
    reports_path.write_text("".join(report_text_blocks), encoding="utf-8")
    print(f"[OK] Classification reports completos de dev guardados en: {reports_path}")

    cm_path = REPORTES_DIR / "confusion_matrices.txt"
    cm_path.write_text("".join(cm_text_blocks), encoding="utf-8")
    print(f"[OK] Matrices de confusión guardadas en: {cm_path}")

    return {
        "mejores": mejores_por_algoritmo,
        "campeon": campeon_exp,
        "classes": classes,
    }


if __name__ == "__main__":
    ejecutar_experimentos()

