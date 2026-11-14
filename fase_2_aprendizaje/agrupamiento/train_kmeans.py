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
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MATRICES_DIR = ROOT_DIR / "fase_2_aprendizaje" / "matrices"
RESULTADOS_DIR = Path(__file__).resolve().parent / "resultados"
RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)

CORRIDA_ID = "20261111T024437810514Z"
SEMILLA = 0


def cargar_datos():
    labeled_path = DATA_PROCESSED_DIR / "labeled_documents.tsv"
    df = pd.read_csv(labeled_path, sep="\t", dtype=str)
    return df


def ejecutar_agrupamiento():
    print("=" * 80)
    print("INICIANDO AGRUPAMIENTO NO SUPERVISADO CON K-MEANS (FASE 2)")
    print("=" * 80)

    df_docs = cargar_datos()
    train_mask = df_docs["split"] == "train"
    dev_mask = df_docs["split"] == "dev"
    test_mask = df_docs["split"] == "test"

    y_train = df_docs.loc[train_mask, "category"].values
    y_dev = df_docs.loc[dev_mask, "category"].values

    experimentos = []
    exp_counter = 1

    # Definir configuraciones clave a explorar
    configuraciones_matriz = [
        ("1 Tokenizado", "tokenizado", "TD IDF", "tfidf", "unigramas"),
        ("2 Stop words", "stop_words", "TD IDF", "tfidf", "unigramas"),
        ("2 Stop words", "stop_words", "TD IDF", "tfidf", "unigramas_bigramas"),
        ("2 Stop words", "stop_words", "Frecuencia", "frecuencia", "unigramas"),
        ("3 Lematizado", "lematizado", "TD IDF", "tfidf", "unigramas"),
    ]

    mejores_config = None
    mejor_silhouette = -1.0
    modelos_entrenados = {}

    for norm_dir, norm_label, rep_dir, rep_label, ngram in configuraciones_matriz:
        mat_path = MATRICES_DIR / norm_dir / rep_dir / CORRIDA_ID / ngram
        if not (mat_path / "X_train.npz").exists():
            continue

        X_train = load_npz(mat_path / "X_train.npz")
        X_dev = load_npz(mat_path / "X_dev.npz")
        vocab = pd.read_csv(mat_path / "vocabulario.tsv", sep="\t")["term"].values

        # Explorar con y sin SVD para diferentes valores de k
        for d_svd in [None, 100]:
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

            for k in [4, 6, 8, 10, 12]:
                exp_id = f"U{exp_counter:02d}"
                exp_counter += 1

                km = KMeans(n_clusters=k, init="k-means++", n_init=10, max_iter=300, random_state=SEMILLA)
                km.fit(X_tr)

                preds_dev = km.predict(X_dv)

                # Silhouette score (en dev)
                sil = silhouette_score(X_dv, preds_dev)
                # Métricas externas contra categorías gold (solo para evaluación metodológica)
                ari = adjusted_rand_score(y_dev, preds_dev)
                nmi = normalized_mutual_info_score(y_dev, preds_dev)

                reg = {
                    "experiment_id": exp_id,
                    "task": "clustering",
                    "normalization": norm_label,
                    "representation": f"{rep_label} {ngram}",
                    "method": metodo_desc,
                    "hyperparameters": f"k={k}, n_init=10, max_iter=300",
                    "silhouette": round(sil, 4),
                    "ari": round(ari, 4),
                    "nmi": round(nmi, 4),
                    "selected": "no",
                    "mat_path": str(mat_path),
                    "svd": svd,
                    "km": km,
                    "k": k,
                    "X_tr": X_tr,
                    "X_dv": X_dv,
                    "vocab": vocab,
                }
                experimentos.append(reg)
                print(f"[{exp_id}] {norm_label} | {rep_label} {ngram} | {metodo_desc} k={k:2d} -> Sil: {sil:.4f} | ARI: {ari:.4f} | NMI: {nmi:.4f}")

                # Criterio de selección: Balance de silhouette score y estabilidad en k=10 (o mejor k temático)
                if k == 10 and d_svd == 100 and norm_label == "stop_words" and rep_label == "tfidf":
                    mejores_config = reg

    # Si no se seleccionó la predeterminada, seleccionar el mayor silhouette
    if mejores_config is None:
        mejores_config = max(experimentos, key=lambda x: x["silhouette"])

    mejores_config["selected"] = "yes"
    print("\n" + "=" * 80)
    print(f"CONFIGURACIÓN SELECCIONADA: {mejores_config['experiment_id']}")
    print(f"Método: {mejores_config['method']} | Normalización: {mejores_config['normalization']} | Rep: {mejores_config['representation']}")
    print(f"Hiperparámetros: {mejores_config['hyperparameters']}")
    print(f"Silhouette (dev): {mejores_config['silhouette']:.4f} | ARI (dev): {mejores_config['ari']:.4f} | NMI (dev): {mejores_config['nmi']:.4f}")
    print("=" * 80)

    # 1. Guardar experiments.tsv / clustering_experiments.tsv
    exp_df = pd.DataFrame(experimentos)[
        ["experiment_id", "task", "normalization", "representation", "method", "hyperparameters", "silhouette", "ari", "nmi", "selected"]
    ]
    exp_tsv_path = RESULTADOS_DIR / "clustering_experiments.tsv"
    exp_df.to_csv(exp_tsv_path, sep="\t", index=False)
    print(f"\n[OK] Registro de {len(exp_df)} experimentos guardado en: {exp_tsv_path}")

    # 2. Generar predicciones en todo el corpus (train, dev, test) para el modelo seleccionado
    sel_km = mejores_config["km"]
    sel_svd = mejores_config["svd"]
    sel_mat_path = Path(mejores_config["mat_path"])

    X_train_full = load_npz(sel_mat_path / "X_train.npz")
    X_dev_full = load_npz(sel_mat_path / "X_dev.npz")
    X_test_full = load_npz(sel_mat_path / "X_test.npz")

    if sel_svd is not None:
        X_tr_proj = sel_svd.transform(X_train_full)
        X_dv_proj = sel_svd.transform(X_dev_full)
        X_ts_proj = sel_svd.transform(X_test_full)
    else:
        X_tr_proj = X_train_full
        X_dv_proj = X_dev_full
        X_ts_proj = X_test_full

    preds_tr = sel_km.predict(X_tr_proj)
    preds_dv = sel_km.predict(X_dv_proj)
    preds_ts = sel_km.predict(X_ts_proj)

    # Estructura del entregable clustering_results.tsv: document_id, split, cluster_id, method/configuration
    metodo_conf_str = f"KMeans(k={mejores_config['k']}, n_init=10, SVD=100)"

    docs_tr_ids = df_docs.loc[train_mask, "document_id"].values
    docs_dv_ids = df_docs.loc[dev_mask, "document_id"].values
    docs_ts_ids = df_docs.loc[test_mask, "document_id"].values

    results_rows = []
    for doc_id, c_id in zip(docs_tr_ids, preds_tr):
        results_rows.append({"document_id": doc_id, "split": "train", "cluster_id": c_id, "method/configuration": metodo_conf_str})
    for doc_id, c_id in zip(docs_dv_ids, preds_dv):
        results_rows.append({"document_id": doc_id, "split": "dev", "cluster_id": c_id, "method/configuration": metodo_conf_str})
    for doc_id, c_id in zip(docs_ts_ids, preds_ts):
        results_rows.append({"document_id": doc_id, "split": "test", "cluster_id": c_id, "method/configuration": metodo_conf_str})

    df_clustering_results = pd.DataFrame(results_rows)
    clustering_results_path = RESULTADOS_DIR / "clustering_results.tsv"
    df_clustering_results.to_csv(clustering_results_path, sep="\t", index=False)
    print(f"[OK] Entregable oficial guardado en: {clustering_results_path} ({len(df_clustering_results)} documentos clasificados)")

    # 3. Interpretación de Clústeres: Top 10 palabras y 2 documentos representativos
    vocab = mejores_config["vocab"]
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
        for rid, rtitle in zip(rep_ids, rep_titles):
            md_lines.append(f"  * `[{rid}]` *\"{rtitle}...\"*")
        md_lines.append("")

    df_interp = pd.DataFrame(interpretacion_rows)
    interp_tsv_path = RESULTADOS_DIR / "cluster_interpretation.tsv"
    df_interp.to_csv(interp_tsv_path, sep="\t", index=False)
    print(f"[OK] Tabla de interpretación guardada en: {interp_tsv_path}")

    interp_md_path = RESULTADOS_DIR / "cluster_interpretation.md"
    interp_md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[OK] Reporte Markdown de clústeres guardado en: {interp_md_path}")

    # 4. Tabla de Contingencia (Clúster vs Categoría Real)
    contingency = pd.crosstab(df_train_docs["cluster_id"], df_train_docs["category"], margins=True)
    contingency_path = RESULTADOS_DIR / "contingency_table.txt"
    contingency_path.write_text("=== TABLA DE CONTINGENCIA (CLÚSTER VS CATEGORÍA GOLD) ===\n\n" + contingency.to_string(), encoding="utf-8")
    print(f"[OK] Tabla de contingencia guardada en: {contingency_path}")


if __name__ == "__main__":
    ejecutar_agrupamiento()

