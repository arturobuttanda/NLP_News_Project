import re
import sys
import time
from pathlib import Path
import pandas as pd
import spacy

sys.stdout.reconfigure(encoding='utf-8')

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent.parent
raw_path = ROOT_DIR / "data" / "raw" / "news_raw_corpus_2157.tsv"
output_path = CURRENT_DIR / "corpus_normalizado" / "news_normalized_corpus.tsv"

print("=" * 60)
print("INICIANDO PIPELINE DE NORMALIZACIÓN DE CORPUS (INGLÉS)")
print("=" * 60)

# 1. Cargar modelo de spaCy para inglés
start_time = time.time()
print("\n[1/6] Cargando modelo en_core_web_sm de spaCy...")
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
print(f"Modelo cargado exitosamente en {time.time() - start_time:.2f}s.")
print(f"Componentes activos en pipeline: {nlp.pipe_names}")

# 2. Ingesta del corpus crudo
print(f"\n[2/6] Leyendo corpus crudo desde: {raw_path}")
df = pd.read_csv(raw_path, sep='\t', encoding='utf-8')
initial_count = len(df)
print(f"Total de registros ingestados: {initial_count}")
print(f"Columnas: {list(df.columns)}")

# 3. Función de pre-limpieza de caracteres
def pre_clean_text(text):
    if not text or pd.isnull(text):
        return ""
    text = str(text)
    # Estandarizar apóstrofes y comillas tipográficas anglosajonas
    text = re.sub(r"[’‘´`]", "'", text)
    text = re.sub(r'[“”]', '"', text)
    # Eliminar emojis y caracteres suplementarios Unicode (plano 1 en adelante)
    text = re.sub(r'[\U00010000-\U0010ffff]', '', text)
    # Reemplazar saltos de línea, retornos y tabuladores por un espacio simple
    text = re.sub(r'[\r\n\t]+', ' ', text)
    # Eliminar caracteres de control no imprimibles
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    # Eliminar fragmentos de navegacion y suscripcion agregados por algunos feeds.
    boilerplate_patterns = [
        r'\s*continue reading\.{0,3}\s*',
        r'\s*read more(?: on [^.!?]+)?\s*(?:→|\.{0,3})?\s*',
        r'\s*sign up(?: here)?[^.!?]*(?:[.!?]|$)',
        r'\s*subscribe to[^.!?]*(?:[.!?]|$)',
        r'\s*follow us(?: on [^.!?]+)?[.!?]?\s*',
        r'\s*click here(?: to [^.!?]+)?[.!?]?\s*',
        r'\s*read the full (?:article|story)[^.!?]*[.!?]?\s*',
        r'\s*support the guardian[^.!?]*[.!?]?\s*',
        r'\s*get our newsletter[^.!?]*[.!?]?\s*',
        r'\s*delivered to your inbox[^.!?]*[.!?]?\s*',
    ]
    for pattern in boilerplate_patterns:
        text = re.sub(pattern, ' ', text, flags=re.IGNORECASE)
    # Colapsar espacios múltiples
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# 4. Definición de categorías gramaticales a descartar (Stopwords POS)
STOPWORD_POS_TAGS = {
    "DET",
    "ADP",
    "CCONJ",
    "SCONJ",
    "PRON",
    "PART",
    "AUX",
}

def process_doc_tokens(doc):
    normalized_tokens = []
    for token in doc:
        # Filtrar categorías gramaticales funcionales
        if token.pos_ in STOPWORD_POS_TAGS:
            continue
        # Filtrar puntuación y espacios
        if token.pos_ in {"PUNCT", "SPACE"} or token.is_punct or token.is_space:
            continue
        
        # Lematización y conversión a minúsculas
        lemma = token.lemma_.lower().strip()
        
        # Limpiar signos de puntuación residuales adjuntos a los bordes
        lemma = re.sub(r'^[^\w]+|[^\w]+$', '', lemma)
        
        # Descartar lemas vacíos, puramente numéricos o símbolos
        if lemma and not lemma.isspace():
            normalized_tokens.append(lemma)
            
    return " ".join(normalized_tokens)

# Función para procesar una serie en lotes con nlp.pipe
def batch_normalize_texts(series, batch_size=256):
    pre_cleaned = [pre_clean_text(t) for t in series]
    docs = nlp.pipe(pre_cleaned, batch_size=batch_size)
    return [process_doc_tokens(doc) for doc in docs]

# 5. Normalización en lote de 'title' y 'description'
print("\n[3/6] Normalizando titulares ('title') con nlp.pipe...")
t0 = time.time()
df['title'] = batch_normalize_texts(df['title'], batch_size=256)
print(f"Titulares normalizados en {time.time() - t0:.2f}s.")

print("\n[4/6] Normalizando descripciones ('description') con nlp.pipe...")
t1 = time.time()
df['description'] = batch_normalize_texts(df['description'], batch_size=256)
print(f"Descripciones normalizadas en {time.time() - t1:.2f}s.")

# Asegurar integridad en todas las columnas categóricas
for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].astype(str).apply(lambda x: re.sub(r'[\r\n\t]+', ' ', x).strip())

# 6. Exportación a nuevo archivo TSV
print(f"\n[5/6] Guardando corpus normalizado en: {output_path}")
df.to_csv(output_path, sep='\t', index=False, encoding='utf-8')
print("Archivo guardado exitosamente.")
print(f"Registros finales: {len(df)}")

# 7. Muestras de Verificación (Antes vs Después)
print("\n[6/6] Muestras de Verificación (Original vs Normalizado):")
raw_df = pd.read_csv(raw_path, sep='\t', encoding='utf-8')

sample_indices = [0, 1, 2, 45, 100]
for idx in sample_indices:
    if idx < len(df):
        row_norm = df.iloc[idx]
        row_raw = raw_df.iloc[idx]
        print("\n" + "-" * 55)
        print(f"[ID: {row_norm['id']} | Fuente: {row_norm['source']} | Cat: {row_norm['category']}]")
        print("TÍTULO RAW:  ", row_raw['title'])
        print("TÍTULO NORM: ", row_norm['title'])
        print("DESC RAW:    ", str(row_raw['description'])[:120], "...")
        print("DESC NORM:   ", str(row_norm['description'])[:120], "...")

print("\n" + "=" * 60)
print(f"PIPELINE COMPLETADO EXITOSAMENTE EN {time.time() - start_time:.2f} SEGUNDOS.")
print("=" * 60)
