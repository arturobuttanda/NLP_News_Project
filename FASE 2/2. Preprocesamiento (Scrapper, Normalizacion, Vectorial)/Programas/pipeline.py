"""Punto de entrada para adquirir, normalizar y vectorizar noticias."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from sklearn.model_selection import train_test_split

from scrapper import CORPUS_ORIGINAL, ampliar_corpus, cargar_corpus
from ngramas import VARIANTES, guardar_ngramas
from representaciones import SPLITS, guardar_representaciones


SEMILLA = 0
RESULTADOS = Path(__file__).resolve().parents[1] / "resultados"
CARPETAS_NORMALIZACION = {
    "minima": "1. Combinado/1. Ind Tokenizado",
    "limpieza": "1. Combinado/1. Ind Tokenizado",
    "combinado_stopwords": "1. Combinado/1.1 Comb Stop words",
    "combinado_lemas": "1. Combinado/1.2 Comb Lematizado",
    "tokenizado_lemas": "1. Combinado/1.3 Tokenizado Lematizado",
    "stopwords_lemas": "1. Combinado/1.4 Stop words Lematizado",
    "sin_stopwords": "2. Individual/2. Ind Stop words",
    "lemas": "2. Individual/3. Ind Lematizado",
}
CARPETAS_REPRESENTACION = {"binaria": "Binario", "frecuencia": "Frecuencia", "tfidf": "TD IDF"}


def argumentos():
    """Define y valida las opciones de ejecución del pipeline."""
    # Se configuran las opciones de entrada y las etapas disponibles.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus", type=Path, default=CORPUS_ORIGINAL,
        help="TSV de entrada (por defecto: Corpus\\news_raw_corpus.tsv).",
    )
    parser.add_argument("--recolectar", action="store_true", help="Añadir noticias RSS al corpus de entrada.")
    parser.add_argument("--dias", type=int, default=14, help="Ventana RSS en días (por defecto: 14).")
    args = parser.parse_args()
    if args.dias < 1:
        parser.error("--dias debe ser un entero positivo.")
    return args



def preparar_documentos(corpus):
    """Construye el texto y asigna las particiones supervisadas."""
    # Se combinan título y descripción para formar cada documento.
    documentos = corpus.rename(columns={"id": "document_id"})
    documentos["text"] = (documentos["title"] + " " + documentos["description"]).str.strip()
    etiquetas = documentos["category"]
    # 70% para train; el 30% restante se divide al 50% para dev y test (15% cada uno).
    # Primero se separa el 70% destinado a entrenamiento.
    train, restante = train_test_split(
        documentos.index, test_size=0.30, shuffle=True, random_state=SEMILLA,
        stratify=etiquetas,
    )
    # El 30% restante se divide por igual entre dev y test.
    dev, test = train_test_split(
        restante, test_size=0.50, shuffle=True, random_state=SEMILLA,
        stratify=etiquetas.loc[restante],
    )
    documentos.loc[train, "split"] = "train"
    documentos.loc[dev, "split"] = "dev"
    documentos.loc[test, "split"] = "test"
    return documentos[["document_id", "text", "category", "split", "source", "date", "url", "retrieved_at"]]


def vectorizar_corrida():
    """Vectoriza las siete variantes de n-gramas preparadas."""
    # Se usa la primera tabla como referencia de documentos y particiones.
    referencia = None
    # Se devuelven las carpetas de representación generadas.
    destinos = []
    # Todas las variantes deben compartir los mismos documentos.
    for modo in VARIANTES.values():
        carpeta = CARPETAS_NORMALIZACION[modo]
        origen = RESULTADOS / carpeta / "N-gramas"
        # Se cargan las etiquetas y particiones de la variante actual.
        documentos = pd.read_csv(origen / "labeled_documents.tsv", sep="\t", dtype=str, keep_default_na=False)
        if set(documentos["split"]) != set(SPLITS) or not documentos["document_id"].is_unique:
            raise ValueError(f"Las particiones deben estar completas y los IDs deben ser únicos: {origen}")
        if referencia is not None and not documentos.equals(referencia):
            raise ValueError(f"Las variantes deben compartir los documentos y sus particiones: {origen}")
        # Esta tabla sirve para comparar las siguientes variantes.
        referencia = documentos

        print(f"Vectorizando {carpeta}: 7 combinaciones × 3 representaciones.")
        guardar_representaciones(origen, RESULTADOS / carpeta, CARPETAS_REPRESENTACION, documentos)
        for representacion in CARPETAS_REPRESENTACION.values():
            destinos.append(RESULTADOS / carpeta / representacion)
    return destinos


def ejecutar(args):
    """Ejecuta en una sola corrida la adquisición y todo el procesamiento."""
    # Se garantiza que exista el directorio general de resultados.
    RESULTADOS.mkdir(exist_ok=True)
    with TemporaryDirectory(prefix=".procesando-", dir=RESULTADOS) as temporal:
        trabajo = Path(temporal)
        # Se carga el corpus y opcionalmente se amplía con RSS.
        corpus = cargar_corpus(args.corpus)
        print(f"Corpus de entrada: {len(corpus)} noticias.")
        if args.recolectar:
            corpus = ampliar_corpus(corpus, trabajo / "rss.db", args.dias)
        # Se asignan texto y particiones antes de cualquier transformación.
        documentos = preparar_documentos(corpus)
        # Se preparan los n-gramas para las siete variantes.
        for modo in VARIANTES.values():
            destino = RESULTADOS / CARPETAS_NORMALIZACION[modo] / "N-gramas"
            destino.mkdir(parents=True, exist_ok=True)
            corpus.to_csv(destino / "news_raw_corpus.tsv", sep="\t", index=False)
            documentos.to_csv(destino / "labeled_documents.tsv", sep="\t", index=False)
            guardar_ngramas(documentos, destino, modo)

        # Se vectorizan todas las variantes y sus representaciones.
        return vectorizar_corrida()


if __name__ == "__main__":
    try:
        for destino in ejecutar(argumentos()):
            print(f"Resultados: {destino}")
    except (OSError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from None
