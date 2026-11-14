import os
import sys
import time
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.sparse import issparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

sys.stdout.reconfigure(encoding='utf-8')

# Definición de rutas del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = BASE_DIR.parent
NORMALIZATION_DIR = PROJECT_DIR / "normalizacion"
DATA_PATH = NORMALIZATION_DIR / "corpus_normalizado" / "news_normalized_corpus.tsv"
MODELS_DIR = BASE_DIR / "resultados" / "models"

print("=" * 70)
print("7. VECTOR SPACE MODEL REPRESENTATION PIPELINE")
print("=" * 70)
print(f"Cargando corpus procesado desde: {DATA_PATH}")

if not DATA_PATH.exists():
    raise FileNotFoundError(f"No se encontró el archivo de datos procesados en {DATA_PATH}")

df = pd.read_csv(DATA_PATH, sep='\t', encoding='utf-8')
print(f"Total de documentos en el corpus: {len(df)}")
print(f"Columnas disponibles: {list(df.columns)}")

# Campos a procesar según la especificación: Title y Description/Body
TARGET_FIELDS = ["title", "description"]

# Configuraciones de representación solicitadas: Unigramas y Bigramas (1, 2)
REPRESENTATIONS = {
    "frequency": {
        "name": "Frequency (Count)",
        "vectorizer": CountVectorizer(ngram_range=(1, 2), binary=False)
    },
    "binary": {
        "name": "Binary (One-Hot)",
        "vectorizer": CountVectorizer(ngram_range=(1, 2), binary=True)
    },
    "tfidf": {
        "name": "TF-IDF",
        "vectorizer": TfidfVectorizer(ngram_range=(1, 2))
    }
}

summary_records = []

for field in TARGET_FIELDS:
    print("\n" + "#" * 70)
    print(f"PROCESANDO CAMPO: '{field.upper()}'")
    print("#" * 70)
    
    # Asegurar tipo string y manejo de posibles nulos
    corpus = df[field].fillna("").astype(str).tolist()
    field_dir = MODELS_DIR / field
    field_dir.mkdir(parents=True, exist_ok=True)
    
    for rep_key, rep_info in REPRESENTATIONS.items():
        t0 = time.time()
        print(f"\n--> Generando representación: {rep_info['name']} (Unigramas + Bigramas)")
        
        # Instanciar un vectorizador limpio para cada iteración
        if rep_key == "frequency":
            vec = CountVectorizer(ngram_range=(1, 2), binary=False)
        elif rep_key == "binary":
            vec = CountVectorizer(ngram_range=(1, 2), binary=True)
        elif rep_key == "tfidf":
            vec = TfidfVectorizer(ngram_range=(1, 2))
            
        # Ajustar y transformar la matriz dispersa
        matrix = vec.fit_transform(corpus)
        duration = time.time() - t0
        
        # Extracción de metadatos del vocabulario
        feature_names = vec.get_feature_names_out()
        n_samples, n_features = matrix.shape
        
        # Conteo de unigramas vs bigramas
        unigrams_count = sum(1 for feat in feature_names if " " not in feat)
        bigrams_count = n_features - unigrams_count
        
        # Cálculo de dispersión (sparsity)
        total_elements = n_samples * n_features
        non_zeros = matrix.nnz
        sparsity = (1.0 - (non_zeros / total_elements)) * 100.0
        
        print(f"    - Dimensión de la matriz: {n_samples} documentos x {n_features:,} features")
        print(f"    - Desglose de vocabulario: {unigrams_count:,} unigramas | {bigrams_count:,} bigramas")
        print(f"    - Elementos no nulos: {non_zeros:,} (Dispersión: {sparsity:.4f}%)")
        print(f"    - Tiempo de vectorización: {duration:.3f}s")
        
        # Guardar en disco en formato PKL
        # 1. Guardar vectorizador individual
        vec_path = field_dir / f"{field}_{rep_key}_vectorizer.pkl"
        with open(vec_path, "wb") as f:
            pickle.dump(vec, f)
            
        # 2. Guardar matriz dispersa individual
        mat_path = field_dir / f"{field}_{rep_key}_matrix.pkl"
        with open(mat_path, "wb") as f:
            pickle.dump(matrix, f)
            
        # 3. Guardar paquete integrado (Bundle)
        bundle_path = field_dir / f"{field}_{rep_key}_unigram_bigram.pkl"
        bundle = {
            "field": field,
            "representation": rep_key,
            "representation_name": rep_info["name"],
            "ngram_range": (1, 2),
            "vectorizer": vec,
            "matrix": matrix,
            "feature_names": feature_names,
            "shape": matrix.shape,
            "sparsity": sparsity,
            "n_unigrams": unigrams_count,
            "n_bigrams": bigrams_count
        }
        with open(bundle_path, "wb") as f:
            pickle.dump(bundle, f)
            
        print(f"    [OK] Archivos guardados en: {field_dir}")
        print(f"         - Vectorizador: {vec_path.name} ({vec_path.stat().st_size / 1024:.1f} KB)")
        print(f"         - Matriz:       {mat_path.name} ({mat_path.stat().st_size / 1024:.1f} KB)")
        print(f"         - Paquete PKL:  {bundle_path.name} ({bundle_path.stat().st_size / 1024:.1f} KB)")
        
        # Muestra de términos más relevantes
        if rep_key in ["frequency", "binary"]:
            sums = np.array(matrix.sum(axis=0)).flatten()
            top_indices = sums.argsort()[::-1][:8]
            top_terms = [(feature_names[i], int(sums[i])) for i in top_indices]
            print(f"    - Top 8 términos/bigramas por suma: {top_terms}")
        elif rep_key == "tfidf":
            means = np.array(matrix.mean(axis=0)).flatten()
            top_indices = means.argsort()[::-1][:8]
            top_terms = [(feature_names[i], round(float(means[i]), 4)) for i in top_indices]
            print(f"    - Top 8 términos/bigramas por TF-IDF medio: {top_terms}")
            
        summary_records.append({
            "Field": field,
            "Representation": rep_info["name"],
            "Matrix Shape": f"{n_samples} x {n_features:,}",
            "Unigrams": f"{unigrams_count:,}",
            "Bigrams": f"{bigrams_count:,}",
            "Sparsity": f"{sparsity:.2f}%"
        })

# Resumen final en tabla
print("\n" + "=" * 70)
print("RESUMEN GENERAL DE MATRICES VECTORIALES GENERADAS (PUNTO 7)")
print("=" * 70)
summary_df = pd.DataFrame(summary_records)
print(summary_df.to_string(index=False))

# Verificación de integridad de carga
print("\n" + "=" * 70)
print("VERIFICACIÓN DE INTEGRIDAD DE ARCHIVOS .PKL")
print("=" * 70)
for field in TARGET_FIELDS:
    for rep_key in ["frequency", "binary", "tfidf"]:
        pkl_test_path = MODELS_DIR / field / f"{field}_{rep_key}_unigram_bigram.pkl"
        with open(pkl_test_path, "rb") as f:
            loaded_data = pickle.load(f)
        assert issparse(loaded_data["matrix"]), "La matriz cargada no es dispersa"
        print(f"[VERIFICADO OK] {pkl_test_path.name} cargado correctamente. Matriz: {loaded_data['matrix'].shape}")

print("\nPipeline de Vector Space Model finalizado exitosamente.")
