import re
import sys
import time
import pickle
import argparse
from pathlib import Path
import pandas as pd
import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.stdout.reconfigure(encoding='utf-8')

# Definición de rutas base
BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = BASE_DIR.parent
ROOT_DIR = PROJECT_DIR.parent
NORMALIZATION_DIR = PROJECT_DIR / "normalizacion"
DATA_PATH = NORMALIZATION_DIR / "corpus_normalizado" / "news_normalized_corpus.tsv"
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "news_raw_corpus_2157.tsv"
MODELS_DIR = BASE_DIR / "resultados" / "models"

# 1. Cargar spaCy (modelo inglés optimizado) para normalización de consultas
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
STOPWORD_POS_TAGS = {"DET", "ADP", "CCONJ", "SCONJ", "PRON", "PART", "AUX"}

def pre_clean_text(text):
    if not text or pd.isnull(text):
        return ""
    text = str(text)
    text = re.sub(r"[’‘´`]", "'", text)
    text = re.sub(r'[“”]', '"', text)
    text = re.sub(r'[\U00010000-\U0010ffff]', '', text)
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def normalize_query(query_text):
    """
    Aplica exactamente el mismo proceso de normalización usado en el corpus.
    """
    cleaned = pre_clean_text(query_text)
    if not cleaned:
        return ""
    doc = nlp(cleaned)
    tokens = []
    for token in doc:
        if token.pos_ in STOPWORD_POS_TAGS:
            continue
        if token.pos_ in {"PUNCT", "SPACE"} or token.is_punct or token.is_space:
            continue
        lemma = token.lemma_.lower().strip()
        lemma = re.sub(r'^[^\w]+|[^\w]+$', '', lemma)
        if lemma and not lemma.isspace():
            tokens.append(lemma)
    return " ".join(tokens)

# 2. Clase Gestora de Modelos y Búsqueda (Information Retrieval Engine)
class InformationRetrievalEngine:
    def __init__(self, data_path=DATA_PATH, raw_data_path=RAW_DATA_PATH, models_dir=MODELS_DIR):
        self.data_path = Path(data_path)
        self.raw_data_path = Path(raw_data_path)
        self.models_dir = Path(models_dir)
        
        # Cargar datasets procesado y crudo
        self.df_proc = pd.read_csv(self.data_path, sep='\t', encoding='utf-8')
        if self.raw_data_path.exists():
            self.df_raw = pd.read_csv(self.raw_data_path, sep='\t', encoding='utf-8')
        else:
            self.df_raw = self.df_proc.copy()
            
        # Diccionario en memoria para vectorizadores y matrices
        self.cache = {}

    def _get_ngram_range(self, feature):
        feat = feature.lower().strip()
        if feat in ["unigram", "unigrams", "1"]:
            return (1, 1), "unigram"
        elif feat in ["bigram", "bigrams", "2"]:
            return (2, 2), "bigram"
        elif feat in ["both", "unigram_bigram", "unigrams and bigrams", "1-2", "1_2"]:
            return (1, 2), "unigram_bigram"
        else:
            raise ValueError(f"Feature inválida: '{feature}'. Opciones válidas: 'unigram', 'bigram', 'both'")

    def _get_or_create_model(self, content, feature_key, representation):
        content = content.lower().strip()
        rep = representation.lower().strip()
        ngram_range, feat_name = self._get_ngram_range(feature_key)
        
        cache_key = f"{content}_{feat_name}_{rep}"
        if cache_key in self.cache:
            return self.cache[cache_key]["vectorizer"], self.cache[cache_key]["matrix"]

        field_dir = self.models_dir / content
        field_dir.mkdir(parents=True, exist_ok=True)
        bundle_path = field_dir / f"{content}_{rep}_{feat_name}.pkl"

        # Intentar cargar desde archivo PKL si ya existe
        if bundle_path.exists():
            with open(bundle_path, "rb") as f:
                bundle = pickle.load(f)
            self.cache[cache_key] = bundle
            return bundle["vectorizer"], bundle["matrix"]

        # Si no existe, construir y entrenar
        corpus = self.df_proc[content].fillna("").astype(str).tolist()
        if rep == "frequency":
            vec = CountVectorizer(ngram_range=ngram_range, binary=False)
        elif rep == "binary":
            vec = CountVectorizer(ngram_range=ngram_range, binary=True)
        elif rep in ["tfidf", "tf-idf"]:
            vec = TfidfVectorizer(ngram_range=ngram_range)
        else:
            raise ValueError(f"Representación no soportada: '{rep}'. Opciones: 'frequency', 'binary', 'tfidf'")

        mat = vec.fit_transform(corpus)
        bundle = {
            "field": content,
            "representation": rep,
            "ngram_range": ngram_range,
            "vectorizer": vec,
            "matrix": mat,
            "feature_names": vec.get_feature_names_out(),
            "shape": mat.shape
        }

        # Guardar PKL
        with open(bundle_path, "wb") as f:
            pickle.dump(bundle, f)

        self.cache[cache_key] = bundle
        return vec, mat

    def search(self, query, content="title", feature="unigram", representation="tfidf", top_k=10):
        """
        Ejecuta la recuperación de información para una consulta en texto libre:
        1. Normaliza la consulta con el mismo pipeline.
        2. Vectoriza la consulta en el espacio vectorial seleccionado.
        3. Calcula la similitud de cosenos frente a la matriz de documentos.
        4. Ordena y retorna los Top K documentos más similares.
        """
        t0 = time.time()
        norm_query = normalize_query(query)
        vec, doc_matrix = self._get_or_create_model(content, feature, representation)
        
        if not norm_query:
            query_vec = vec.transform([""])
        else:
            query_vec = vec.transform([norm_query])

        # Calcular similitud de cosenos
        sim_scores = cosine_similarity(query_vec, doc_matrix).flatten()
        
        # Ordenar de mayor a menor
        sorted_indices = sim_scores.argsort()[::-1]
        
        # Extraer los Top K
        results = []
        for rank, idx in enumerate(sorted_indices[:top_k], start=1):
            score = float(sim_scores[idx])
            row_raw = self.df_raw.iloc[idx]
            row_proc = self.df_proc.iloc[idx]
            
            results.append({
                "rank": rank,
                "score": round(score, 4),
                "id": row_raw["id"],
                "source": row_raw["source"],
                "category": row_raw["category"],
                "date": row_raw["date"],
                "raw_title": row_raw["title"],
                "normalized_title": row_proc["title"],
                "raw_description": row_raw["description"],
                "normalized_description": row_proc["description"],
                "url": row_raw["url"]
            })

        elapsed = time.time() - t0
        return {
            "query": query,
            "normalized_query": norm_query,
            "content": content,
            "feature": feature,
            "representation": representation,
            "vocab_size": doc_matrix.shape[1],
            "total_documents": doc_matrix.shape[0],
            "elapsed_seconds": round(elapsed, 4),
            "results": results
        }

def display_search_results(res):
    print("\n" + "=" * 80)
    print("8. INFORMATION RETRIEVAL - RESULTADOS DE CONSULTA")
    print("=" * 80)
    print(f"Consulta Original:       \"{res['query']}\"")
    print(f"Consulta Normalizada:    \"{res['normalized_query']}\"")
    print(f"Contenido comparativo:   {res['content'].upper()}")
    print(f"Feature (N-gramas):      {res['feature'].upper()}")
    print(f"Tipo Representación:     {res['representation'].upper()} (Vocabulario: {res['vocab_size']:,} términos)")
    print(f"Tiempo de ejecución:     {res['elapsed_seconds']} segundos")
    print("-" * 80)
    
    non_zero = sum(1 for r in res['results'] if r['score'] > 0)
    if non_zero == 0:
        print("[!] No se encontraron documentos con similitud mayor a 0 para los términos de la consulta.")
        print("-" * 80)
        return

    print(f"{'RANK':<5} | {'SCORE':<7} | {'ID':<6} | {'CATEGORÍA':<15} | {'FUENTE':<18} | {'TÍTULO DE LA NOTICIA'}")
    print("-" * 80)
    for r in res['results']:
        title_snippet = (r['raw_title'][:55] + "...") if len(r['raw_title']) > 55 else r['raw_title']
        print(f"#{r['rank']:<4} | {r['score']:<7.4f} | {r['id']:<6} | {r['category']:<15} | {r['source'][:18]:<18} | {title_snippet}")
    
    print("\n--> Detalle del Resultado #1 (Más Relevante):")
    top1 = res['results'][0]
    print(f"    [ID: {top1['id']} | Score: {top1['score']:.4f} | Categoría: {top1['category']} | Fuente: {top1['source']}]")
    print(f"    TÍTULO:       {top1['raw_title']}")
    print(f"    DESCRIPCIÓN:  {top1['raw_description']}")
    print(f"    URL:          {top1['url']}")
    print("=" * 80)

# 3. Función para ejecutar los ejemplos requeridos en la especificación
def run_assignment_examples():
    engine = InformationRetrievalEngine()
    
    examples = [
        {
            "query": "impact of the hurricane on the coast",
            "content": "description",
            "feature": "unigram",
            "representation": "tfidf"
        },
        {
            "query": "artificial intelligence regulation",
            "content": "title",
            "feature": "unigram",
            "representation": "tfidf"
        },
        {
            "query": "results of a technology company",
            "content": "description",
            "feature": "unigram",
            "representation": "frequency"
        },
        {
            "query": "artificial intelligence",
            "content": "title",
            "feature": "bigram",
            "representation": "binary"
        }
    ]

    print("\n" + "#" * 80)
    print("EJECUCIÓN DE CONSULTAS DE PRUEBA Y EJEMPLOS DEL PROYECTO")
    print("#" * 80)
    
    for ex in examples:
        res = engine.search(
            query=ex["query"],
            content=ex["content"],
            feature=ex["feature"],
            representation=ex["representation"],
            top_k=10
        )
        display_search_results(res)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="8. Information Retrieval Engine")
    parser.add_argument("--query", type=str, default="", help="Texto de la consulta")
    parser.add_argument("--content", type=str, default="title", choices=["title", "description"], help="Contenido comparativo")
    parser.add_argument("--feature", type=str, default="unigram", choices=["unigram", "bigram", "both"], help="Extracción de features")
    parser.add_argument("--representation", type=str, default="tfidf", choices=["frequency", "binary", "tfidf"], help="Representación vectorial")
    parser.add_argument("--top_k", type=int, default=10, help="Número de documentos a mostrar")
    parser.add_argument("--run-examples", action="store_true", help="Ejecutar los ejemplos oficiales del documento")
    parser.add_argument("--interactive", action="store_true", help="Modo interactivo por consola")

    args = parser.parse_args()

    if args.run_examples:
        run_assignment_examples()
    elif args.interactive:
        engine = InformationRetrievalEngine()
        print("\n=== MODO INTERACTIVO DE RECUPERACIÓN DE INFORMACIÓN ===")
        print("Escribe 'exit' o 'salir' para terminar.\n")
        while True:
            try:
                q = input("\nIngrese su consulta (en texto libre): ").strip()
                if q.lower() in ["exit", "salir", "q"]:
                    break
                if not q:
                    continue
                cnt = input("Contenido a comparar [1: title (default) / 2: description]: ").strip()
                cnt = "description" if cnt == "2" else "title"
                
                feat = input("Feature [1: unigram (default) / 2: bigram / 3: both]: ").strip()
                if feat == "2":
                    feat = "bigram"
                elif feat == "3":
                    feat = "both"
                else:
                    feat = "unigram"
                    
                rep = input("Representación [1: tfidf (default) / 2: frequency / 3: binary]: ").strip()
                if rep == "2":
                    rep = "frequency"
                elif rep == "3":
                    rep = "binary"
                else:
                    rep = "tfidf"
                    
                res = engine.search(query=q, content=cnt, feature=feat, representation=rep, top_k=10)
                display_search_results(res)
            except KeyboardInterrupt:
                break
    elif args.query:
        engine = InformationRetrievalEngine()
        res = engine.search(
            query=args.query,
            content=args.content,
            feature=args.feature,
            representation=args.representation,
            top_k=args.top_k
        )
        display_search_results(res)
    else:
        # Por defecto si no se especifican argumentos, ejecutar las pruebas oficiales
        run_assignment_examples()
