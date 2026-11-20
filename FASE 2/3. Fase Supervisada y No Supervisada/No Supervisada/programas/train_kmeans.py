"""Agrupamiento no supervisado de noticias con K-Means (Tarea B - Fase 2).

Explora sistemáticamente configuraciones de K-Means variando k, representaciones (TF-IDF vs Count),
normalizaciones y reducción de dimensionalidad con TruncatedSVD.
Evalúa Silhouette score, ARI y NMI. Genera los entregables oficiales en resultados/:
  - clustering_results.tsv
  - cluster_interpretation.tsv y cluster_interpretation.md
  - clustering_experiments.tsv
  - contingency_table.txt
"""

import json
import gc
import re
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

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
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MATRICES_DIR = ROOT_DIR / "fase_2_aprendizaje" / "matrices"
RESULTADOS_DIR = TASK_DIR / "resultados"
EXPERIMENTACION_DIR = RESULTADOS_DIR / "01_experimentacion"
ASIGNACIONES_DIR = RESULTADOS_DIR / "02_asignaciones"
INTERPRETACION_DIR = RESULTADOS_DIR / "03_interpretacion"
for carpeta in [EXPERIMENTACION_DIR, ASIGNACIONES_DIR, INTERPRETACION_DIR]:
    # Se separan los resultados cuantitativos de la interpretación.
    carpeta.mkdir(parents=True, exist_ok=True)

CORRIDA_ID = "20261111T024437810514Z"
SEMILLA = 0


def cargar_datos():
    """Carga las etiquetas y particiones generadas por el preprocesamiento."""
    # El archivo conserva document_id, texto, categoría y split.
    labeled_path = (
        PREPROCESSING_RESULTS_DIR
        / "1. Combinado"
        / "1. Ind Tokenizado"
        / "N-gramas"
        / "labeled_documents.tsv"
    )
    if not labeled_path.exists():
        labeled_path = DATA_PROCESSED_DIR / "labeled_documents.tsv"
    df = pd.read_csv(labeled_path, sep="\t", dtype=str)
    return df


def cargar_resultados_previos(log_path, mejor_silhouette=-1.0):
    """Carga métricas completas de la ejecución anterior hasta U1325."""
    if not log_path.exists():
        return [], mejor_silhouette, None

    norm_reverse = {
        "tokenizado_minimo": "1. Combinado/1. Ind Tokenizado",
        "stop_words": "1. Combinado/1.1 Comb Stop words",
        "combinado_lemas": "1. Combinado/1.2 Comb Lematizado",
        "tokenizado_lemas": "1. Combinado/1.3 Tokenizado Lematizado",
        "stopwords_lemas": "1. Combinado/1.4 Stop words Lematizado",
        "solo_stop_words": "2. Individual/2. Ind Stop words",
        "solo_lematizado": "2. Individual/3. Ind Lematizado",
    }
    rep_dirs = {"binary": "Binario", "frecuencia": "Frecuencia", "tfidf": "TD IDF"}
    patron = re.compile(
        r"^\[(U\d+)\] (.+?) \| (binary|frecuencia|tfidf) "
        r"(unigramas|bigramas|trigramas|unigramas_bigramas|"
        r"unigramas_trigramas|bigramas_trigramas|unigramas_bigramas_trigramas)"
        r" \| (KMeans(?:\+SVD\(\d+\))?) k=(\d+) -> Sil: ([+-]?\d+(?:\.\d+)?) "
        r"\| ARI: ([+-]?\d+(?:\.\d+)?) \| NMI: ([+-]?\d+(?:\.\d+)?)$"
    )
    registros = []
    mejor = None
    for linea in log_path.read_text(encoding="utf-8").splitlines():
        coincidencia = patron.search(linea)
        if not coincidencia:
            continue
        exp_id, norm_label, rep_label, ngram, metodo, k, sil, ari, nmi = coincidencia.groups()
        num = int(exp_id[1:])
        if num > 1325:
            continue

        reg = {
            "experiment_id": exp_id,
            "task": "clustering",
            "configuration_role": "baseline" if norm_label == "tokenizado_minimo" else "processed",
            "normalization": norm_label,
            "representation": f"{rep_label} {ngram}",
            "method": metodo,
            "hyperparameters": f"k={k}, init=k-means++, n_init=10, max_iter=300, random_state=0",
            "silhouette": float(sil),
            "ari": float(ari),
            "nmi": float(nmi),
            "selected": "no",
            "k": int(k),
        }
        norm_dir = norm_reverse.get(norm_label)
        rep_dir = rep_dirs.get(rep_label)
        if norm_dir and rep_dir:
            reg["mat_path"] = str(PREPROCESSING_RESULTS_DIR / norm_dir / rep_dir / ngram)
        registros.append(reg)

        if float(sil) > mejor_silhouette:
            mejor_silhouette = float(sil)
            mejor = dict(reg)
            mejor["k"] = int(k)
            mejor["svd_dim"] = 100 if "+SVD(" in metodo else None
    return registros, mejor_silhouette, mejor


def ejecutar_agrupamiento(resume_from=1, resume_to=1470):
    """Entrena K-Means, compara configuraciones y genera los entregables no supervisados."""
    print("=" * 80)
    print("INICIANDO AGRUPAMIENTO NO SUPERVISADO CON K-MEANS (FASE 2)")
    print("=" * 80)

    df_docs = cargar_datos()
    train_mask = df_docs["split"] == "train"
    dev_mask = df_docs["split"] == "dev"
    test_mask = df_docs["split"] == "test"

    y_train = df_docs.loc[train_mask, "category"].values
    y_dev = df_docs.loc[dev_mask, "category"].values

    log_prev = (
        RESULTADOS_DIR
        / "01_experimentacion"
        / "kmeans_execution_log.txt"
    )
    experimentos, mejor_silhouette, mejores_config = cargar_resultados_previos(log_prev)
    exp_counter = resume_from

    normalizaciones = {
        # Las claves son rutas reales del preprocesamiento.
        "1. Combinado/1. Ind Tokenizado": "tokenizado_minimo",
        "1. Combinado/1.1 Comb Stop words": "stop_words",
        "1. Combinado/1.2 Comb Lematizado": "combinado_lemas",
        "1. Combinado/1.3 Tokenizado Lematizado": "tokenizado_lemas",
        "1. Combinado/1.4 Stop words Lematizado": "stopwords_lemas",
        "2. Individual/2. Ind Stop words": "solo_stop_words",
        "2. Individual/3. Ind Lematizado": "solo_lematizado",
    }
    representaciones = {
        "Binario": "binary",
        "Frecuencia": "frecuencia",
        "TD IDF": "tfidf",
    }
    combinaciones_ngramas = (
        "unigramas",
        "bigramas",
        "trigramas",
        "unigramas_bigramas",
        "unigramas_trigramas",
        "bigramas_trigramas",
        "unigramas_bigramas_trigramas",
    )

    configuraciones_matriz = [
        # Producto cartesiano: siete preprocesamientos × tres representaciones × siete n-gramas.
        (norm_dir, norm_label, rep_dir, rep_label, ngram)
        for norm_dir, norm_label in normalizaciones.items()
        for rep_dir, rep_label in representaciones.items()
        for ngram in combinaciones_ngramas
    ]

    modelos_entrenados = {}

    # Se evalúa cada combinación de normalización, representación y n-gramas.
    for matriz_index, (norm_dir, norm_label, rep_dir, rep_label, ngram) in enumerate(configuraciones_matriz):
        mat_path = PREPROCESSING_RESULTS_DIR / norm_dir / rep_dir / ngram
        if not (mat_path / "X_train.npz").exists():
            continue

        X_train = load_npz(mat_path / "X_train.npz")
        X_dev = load_npz(mat_path / "X_dev.npz")
        vocab = pd.read_csv(mat_path / "vocabulario.tsv", sep="\t")["term"].values

        # Explorar con y sin SVD para diferentes valores de k
        # Se compara el espacio original con una proyección SVD de 100 dimensiones.
        for d_svd in [None, 100]:
            # SVD proyecta el espacio léxico; None conserva las características originales.
            if d_svd is not None:
                svd = TruncatedSVD(n_components=d_svd, random_state=SEMILLA)
                X_tr = svd.fit_transform(X_train)
                X_dv = svd.transform(X_dev)
                metodo_desc = f"KMeans+SVD({d_svd})"
            else:
                svd = None
                X_tr = X_train
                X_dv = X_dev
                metodo_desc = "KMeans"

            # Se prueban distintos tamaños de agrupamiento.
            for k in [4, 6, 8, 10, 12]:
                # Cada k representa una hipótesis distinta sobre el número de grupos.
                if exp_counter > resume_to:
                    break
                exp_id = f"U{exp_counter:02d}"
                exp_counter += 1
                if int(exp_id[1:]) < resume_from:
                    continue

                km = KMeans(n_clusters=k, init="k-means++", n_init=10, max_iter=300, random_state=SEMILLA)
                # Las etiquetas no se pasan a fit: el agrupamiento aprende solo de X_tr.
                km.fit(X_tr)

                preds_dev = km.predict(X_dv)

                # Silhouette requiere al menos dos clústeres y menos que
                # el número total de documentos. Una configuración degenerada
                # se registra sin puntuar Silhouette para continuar el barrido.
                etiquetas_unicas = np.unique(preds_dev)
                if 2 <= len(etiquetas_unicas) < len(preds_dev):
                    sil = silhouette_score(X_dv, preds_dev)
                else:
                    sil = -1.0
                # Métricas externas contra categorías gold (solo para evaluación metodológica)
                # ARI y NMI nunca intervienen en fit ni en la selección intrínseca.
                ari = adjusted_rand_score(y_dev, preds_dev)
                nmi = normalized_mutual_info_score(y_dev, preds_dev)

                reg = {
                    "experiment_id": exp_id,
                    "task": "clustering",
                    "configuration_role": "baseline" if norm_label == "tokenizado_minimo" else "processed",
                    "normalization": norm_label,
                    "representation": f"{rep_label} {ngram}",
                    "method": metodo_desc,
                    "hyperparameters": (
                        f"k={k}, init=k-means++, n_init=10, "
                        f"max_iter=300, random_state={SEMILLA}"
                    ),
                    "silhouette": round(sil, 4),
                    "ari": round(ari, 4),
                    "nmi": round(nmi, 4),
                    "selected": "no",
                    "mat_path": str(mat_path),
                    "k": k,
                }
                experimentos.append(reg)
                print(f"[{exp_id}] {norm_label} | {rep_label} {ngram} | {metodo_desc} k={k:2d} -> Sil: {sil:.4f} | ARI: {ari:.4f} | NMI: {nmi:.4f}")

                if exp_counter % 25 == 0:
                    checkpoint_df = pd.DataFrame(experimentos)[
                        ["experiment_id", "task", "configuration_role", "normalization",
                         "representation", "method", "hyperparameters", "silhouette",
                         "ari", "nmi", "selected"]
                    ]
                    checkpoint_path = EXPERIMENTACION_DIR / "clustering_experiments_checkpoint.tsv"
                    checkpoint_df.to_csv(checkpoint_path, sep="\t", index=False)

                # El mejor Silhouette en dev define la configuración seleccionada.
                if sil > mejor_silhouette:
                    mejor_silhouette = sil
                    mejores_config = reg
                    mejores_config["vocab"] = vocab
                    mejores_config["svd"] = svd
                    mejores_config["km"] = km
                    mejores_config["k"] = k

                del km
                del preds_dev
                gc.collect()

            del X_tr
            del X_dv
            gc.collect()

        del X_train
        del X_dev
        del vocab
        gc.collect()

    if mejores_config is None:
        raise RuntimeError("No se encontró ninguna matriz válida para evaluar K-Means.")

    mejores_config["selected"] = "yes"
    baseline_candidates = [
        exp for exp in experimentos
        if exp["configuration_role"] == "baseline"
    ]
    baseline_comparison = pd.DataFrame()
    if baseline_candidates:
        baseline_exp = max(baseline_candidates, key=lambda exp: exp["silhouette"])
        baseline_comparison = pd.DataFrame([{
            "baseline_role": "only_tokenization",
            "baseline_experiment_id": baseline_exp["experiment_id"],
            "baseline_method": baseline_exp["method"],
            "baseline_representation": baseline_exp["representation"],
            "baseline_k": baseline_exp["k"],
            "baseline_silhouette": baseline_exp["silhouette"],
            "baseline_ari": baseline_exp["ari"],
            "baseline_nmi": baseline_exp["nmi"],
            "champion_experiment_id": mejores_config["experiment_id"],
            "champion_method": mejores_config["method"],
            "champion_representation": mejores_config["representation"],
            "champion_k": mejores_config["k"],
            "champion_silhouette": mejores_config["silhouette"],
            "champion_ari": mejores_config["ari"],
            "champion_nmi": mejores_config["nmi"],
            "silhouette_delta": round(mejores_config["silhouette"] - baseline_exp["silhouette"], 4),
        }])
        baseline_path = EXPERIMENTACION_DIR / "baseline_comparison.tsv"
        baseline_comparison.to_csv(baseline_path, sep="\t", index=False)
        print(f"[OK] Comparación con tokenización mínima guardada en: {baseline_path}")

    print("\n" + "=" * 80)
    print(f"CONFIGURACIÓN SELECCIONADA: {mejores_config['experiment_id']}")
    print(f"Método: {mejores_config['method']} | Normalización: {mejores_config['normalization']} | Rep: {mejores_config['representation']}")
    print(f"Hiperparámetros: {mejores_config['hyperparameters']}")
    print(f"Silhouette (dev): {mejores_config['silhouette']:.4f} | ARI (dev): {mejores_config['ari']:.4f} | NMI (dev): {mejores_config['nmi']:.4f}")
    print("=" * 80)

    # 1. Guardar experiments.tsv / clustering_experiments.tsv
    exp_df = pd.DataFrame(experimentos)[
        ["experiment_id", "task", "configuration_role", "normalization", "representation", "method", "hyperparameters", "silhouette", "ari", "nmi", "selected"]
    ]
    exp_tsv_path = EXPERIMENTACION_DIR / "clustering_experiments.tsv"
    exp_df.to_csv(exp_tsv_path, sep="\t", index=False)
    print(f"\n[OK] Registro de {len(exp_df)} experimentos guardado en: {exp_tsv_path}")

    # 2. Generar predicciones en todo el corpus (train, dev, test) para el modelo seleccionado
    # Se reutilizan los mismos centroides; test no participa en el ajuste.
    sel_mat_path = Path(mejores_config["mat_path"])

    X_train_full = load_npz(sel_mat_path / "X_train.npz")
    X_dev_full = load_npz(sel_mat_path / "X_dev.npz")
    X_test_full = load_npz(sel_mat_path / "X_test.npz")

    sel_svd = None
    if mejores_config.get("svd_dim") is not None:
        sel_svd = TruncatedSVD(n_components=mejores_config["svd_dim"], random_state=SEMILLA)
        X_tr_proj = sel_svd.fit_transform(X_train_full)
        X_dv_proj = sel_svd.transform(X_dev_full)
        X_ts_proj = sel_svd.transform(X_test_full)
    elif mejores_config.get("svd") is not None:
        sel_svd = mejores_config["svd"]
        X_tr_proj = sel_svd.transform(X_train_full)
        X_dv_proj = sel_svd.transform(X_dev_full)
        X_ts_proj = sel_svd.transform(X_test_full)
    else:
        X_tr_proj = X_train_full
        X_dv_proj = X_dev_full
        X_ts_proj = X_test_full

    if mejores_config.get("km") is not None:
        sel_km = mejores_config["km"]
    else:
        sel_km = KMeans(
            n_clusters=mejores_config["k"],
            init="k-means++",
            n_init=10,
            max_iter=300,
            random_state=SEMILLA,
        )
        sel_km.fit(X_tr_proj)

    preds_tr = sel_km.predict(X_tr_proj)
    preds_dv = sel_km.predict(X_dv_proj)
    preds_ts = sel_km.predict(X_ts_proj)

    # Estructura del entregable clustering_results.tsv: document_id, split, cluster_id, method/configuration
    if sel_svd is not None:
        metodo_conf_str = f"KMeans(k={mejores_config['k']}, n_init=10, SVD={mejores_config['method'].split('SVD(')[-1].rstrip(')')})"
    else:
        metodo_conf_str = f"KMeans(k={mejores_config['k']}, n_init=10, SVD=none)"

    docs_tr_ids = df_docs.loc[train_mask, "document_id"].values
    docs_dv_ids = df_docs.loc[dev_mask, "document_id"].values
    docs_ts_ids = df_docs.loc[test_mask, "document_id"].values

    results_rows = []
    for doc_id, c_id in zip(docs_tr_ids, preds_tr):
        # Se asigna a cada documento el clúster aprendido.
        results_rows.append({"document_id": doc_id, "split": "train", "cluster_id": c_id, "method/configuration": metodo_conf_str})
    # Se asignan clústeres a los documentos de desarrollo.
    for doc_id, c_id in zip(docs_dv_ids, preds_dv):
        results_rows.append({"document_id": doc_id, "split": "dev", "cluster_id": c_id, "method/configuration": metodo_conf_str})
    # Se asignan clústeres a los documentos de prueba sin reentrenar.
    for doc_id, c_id in zip(docs_ts_ids, preds_ts):
        results_rows.append({"document_id": doc_id, "split": "test", "cluster_id": c_id, "method/configuration": metodo_conf_str})

    df_clustering_results = pd.DataFrame(results_rows)
    clustering_results_path = ASIGNACIONES_DIR / "clustering_results.tsv"
    df_clustering_results.to_csv(clustering_results_path, sep="\t", index=False)
    print(f"[OK] Entregable oficial guardado en: {clustering_results_path} ({len(df_clustering_results)} documentos clasificados)")

    # 3. Interpretación de Clústeres: Top 10 palabras y 2 documentos representativos
    vocab = mejores_config.get("vocab")
    if vocab is None:
        vocab = pd.read_csv(sel_mat_path / "vocabulario.tsv", sep="\t")["term"].values
    if sel_svd is not None:
        # Proyectar centroides SVD al espacio léxico original: (k, d) @ (d, V) -> (k, V)
        orig_centroids = np.dot(sel_km.cluster_centers_, sel_svd.components_)
    else:
        orig_centroids = sel_km.cluster_centers_

    # Calcular distancias de los documentos de train a sus centroides para hallar los más representativos
    from sklearn.metrics.pairwise import euclidean_distances
    distancias_tr = euclidean_distances(X_tr_proj, sel_km.cluster_centers_)

    k_val = mejores_config["k"]
    interpretacion_rows = []
    md_lines = [
        f"# 🔬 Interpretación de Clústeres - K-Means ({metodo_conf_str})\n",
        f"**Configuración:** {mejores_config['normalization']} | {mejores_config['representation']}\n",
        f"**Silhouette Score:** {mejores_config['silhouette']:.4f} | **NMI:** {mejores_config['nmi']:.4f} | **ARI:** {mejores_config['ari']:.4f}\n",
    ]

    df_train_docs = df_docs[train_mask].reset_index(drop=True)
    df_train_docs["cluster_id"] = preds_tr

    for c in range(k_val):
        # Se resume tamaño, vocabulario y documentos representativos del clúster.
        # Tamaño
        docs_en_c = df_train_docs[df_train_docs["cluster_id"] == c]
        tam = len(docs_en_c)

        # Top 10 palabras del espacio léxico original
        top_word_indices = orig_centroids[c].argsort()[::-1][:10]
        top_words = [vocab[idx] for idx in top_word_indices]
        top_words_str = ", ".join(top_words)

        # 2 documentos más cercanos al centroide (mínima distancia euclidiana en el espacio del modelo)
        indices_cluster = np.where(preds_tr == c)[0]
        if len(indices_cluster) >= 2:
            dists_c = distancias_tr[indices_cluster, c]
            closest_local_indices = dists_c.argsort()[:2]
            rep_indices = indices_cluster[closest_local_indices]
            rep_docs = df_train_docs.iloc[rep_indices]
            rep_ids = list(rep_docs["document_id"].values)
            rep_titles = list(rep_docs["text"].str[:100].values)
        else:
            rep_ids = list(docs_en_c["document_id"].values)
            rep_titles = list(docs_en_c["text"].str[:100].values)

        rep_ids_str = ", ".join(rep_ids)

        # Categoría dominante en el clúster
        cat_counts = docs_en_c["category"].value_counts()
        cat_dominante = cat_counts.index[0] if not cat_counts.empty else "N/A"
        pct_dominante = (cat_counts.iloc[0] / tam * 100) if tam > 0 else 0.0

        interpretacion_rows.append({
            "cluster_id": c,
            "size": tam,
            "top_words": top_words_str,
            "representative_document_ids": rep_ids_str,
            "dominant_category": f"{cat_dominante} ({pct_dominante:.1f}%)",
        })

        md_lines.append(f"### Clúster #{c} (Tamaño: {tam} noticias | Dominante: `{cat_dominante}` {pct_dominante:.1f}%)")
        md_lines.append(f"- **Top 10 Términos:** `{top_words_str}`")
        md_lines.append(f"- **Noticias Representativas:**")
        # Se listan los títulos de los documentos más cercanos al centroide.
        for rid, rtitle in zip(rep_ids, rep_titles):
            md_lines.append(f"  * `[{rid}]` *\"{rtitle}...\"*")
        md_lines.append("")

    df_interp = pd.DataFrame(interpretacion_rows)
    interp_tsv_path = INTERPRETACION_DIR / "cluster_interpretation.tsv"
    df_interp.to_csv(interp_tsv_path, sep="\t", index=False)
    print(f"[OK] Tabla de interpretación guardada en: {interp_tsv_path}")

    interp_md_path = INTERPRETACION_DIR / "cluster_interpretation.md"
    interp_md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[OK] Reporte Markdown de clústeres guardado en: {interp_md_path}")

    # 4. Tabla de Contingencia (Clúster vs Categoría Real)
    contingency = pd.crosstab(df_train_docs["cluster_id"], df_train_docs["category"], margins=True)
    contingency_path = INTERPRETACION_DIR / "contingency_table.txt"
    contingency_path.write_text("=== TABLA DE CONTINGENCIA (CLÚSTER VS CATEGORÍA GOLD) ===\n\n" + contingency.to_string(), encoding="utf-8")
    print(f"[OK] Tabla de contingencia guardada en: {contingency_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-from", type=int, default=1326)
    parser.add_argument("--resume-to", type=int, default=1470)
    args = parser.parse_args()
    ejecutar_agrupamiento(resume_from=args.resume_from, resume_to=args.resume_to)

