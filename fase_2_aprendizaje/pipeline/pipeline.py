"""Punto de entrada para adquirir, normalizar y vectorizar noticias."""

import argparse
import json
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import save_npz
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from scrapper import CORPUS_ORIGINAL, ampliar_corpus, cargar_corpus
from ngramas import VARIANTES, guardar_ngramas
from normalizacion import NormalizadorNoticias
from representaciones import SPLITS, crear_vectorizador, guardar_representaciones


SEMILLA = 0
RESULTADOS = Path(__file__).resolve().parents[1] / "matrices"
CARPETAS_NORMALIZACION = {
    "minima": "1 Tokenizado",
    "limpieza": "1 Tokenizado",
    "sin_stopwords": "2 Stop words",
    "lemas": "3 Lematizado",
}
CARPETAS_REPRESENTACION = {"binaria": "Binario", "frecuencia": "Frecuencia", "tfidf": "TD IDF"}
NGRAMAS = {"unigramas": (1, 1), "bigramas": (2, 2), "ambos": (1, 2)}


def argumentos():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus", type=Path, default=CORPUS_ORIGINAL,
        help="TSV de entrada (por defecto: news_raw_corpus.tsv de esta carpeta).",
    )
    parser.add_argument("--recolectar", action="store_true", help="Añadir noticias RSS al corpus de entrada.")
    parser.add_argument("--dias", type=int, default=14, help="Ventana RSS en días (por defecto: 14).")
    parser.add_argument("--normalizacion", choices=CARPETAS_NORMALIZACION, default="lemas")
    etapa = parser.add_mutually_exclusive_group()
    etapa.add_argument(
        "--preparar-ngramas", action="store_true",
        help="Exportar las tres variantes y sus n-gramas sin vectorizar.",
    )
    etapa.add_argument(
        "--vectorizar-ngramas", metavar="CORRIDA",
        help="Generar las tres representaciones de las siete combinaciones de una corrida de n-gramas.",
    )
    parser.add_argument("--representacion", choices=CARPETAS_REPRESENTACION, default="frecuencia")
    parser.add_argument("--ngramas", choices=NGRAMAS, default="ambos")
    args = parser.parse_args()
    if args.dias < 1:
        parser.error("--dias debe ser un entero positivo.")
    return args



def preparar_documentos(corpus):
    documentos = corpus.rename(columns={"id": "document_id"})
    documentos["text"] = (documentos["title"] + " " + documentos["description"]).str.strip()
    etiquetas = documentos["category"]
    train, restante = train_test_split(
        documentos.index, test_size=0.30, shuffle=True, random_state=SEMILLA,
        stratify=etiquetas,
    )
    dev, test = train_test_split(
        restante, test_size=0.50, shuffle=True, random_state=SEMILLA,
        stratify=etiquetas.loc[restante],
    )
    documentos.loc[train, "split"] = "train"
    documentos.loc[dev, "split"] = "dev"
    documentos.loc[test, "split"] = "test"
    return documentos[["document_id", "text", "category", "split", "source", "date", "url", "retrieved_at"]]


def crear_pipeline(normalizacion, representacion, ngramas):
    return Pipeline([
        ("normalizacion", NormalizadorNoticias(modo=normalizacion)),
        ("vectorizacion", crear_vectorizador(representacion, ngram_range=NGRAMAS[ngramas])),
    ])


def guardar_resultados(corpus, documentos, pipeline, destino, args):
    corpus.to_csv(destino / "news_raw_corpus.tsv", sep="\t", index=False)
    documentos.to_csv(destino / "labeled_documents.tsv", sep="\t", index=False)

    normalizados = documentos[["document_id", "split"]].copy()
    normalizados["text_normalized"] = pipeline["normalizacion"].transform(documentos["text"])
    normalizados.to_csv(destino / "corpus_normalizado.tsv", sep="\t", index=False)

    vectorizador = pipeline["vectorizacion"]
    pd.DataFrame({"term": vectorizador.get_feature_names_out()}).to_csv(
        destino / "vocabulario.tsv", sep="\t", index=False,
    )
    resumen = {}
    for split in SPLITS:
        filas = normalizados.loc[normalizados["split"].eq(split)]
        matriz = vectorizador.transform(filas["text_normalized"]).tocsr()
        save_npz(destino / f"X_{split}.npz", matriz)
        sin_conteos = int(np.count_nonzero(matriz.getnnz(axis=1) == 0))
        resumen[split] = {"filas": matriz.shape[0], "columnas": matriz.shape[1], "vectores_cero": sin_conteos}
        print(f"{split}: {matriz.shape[0]} noticias × {matriz.shape[1]} términos. Vectores cero: {sin_conteos}.")

    joblib.dump(pipeline, destino / "pipeline_vectorial.joblib")
    paquetes = ["scikit-learn", "pandas", "numpy", "scipy", "joblib"]
    if args.normalizacion == "lemas":
        paquetes.extend(["spacy", "en-core-web-sm"])
    if args.recolectar:
        paquetes.extend(["beautifulsoup4", "feedparser", "python-dateutil", "requests"])
    configuracion = {
        **vars(args),
        "corpus": str(args.corpus.resolve()),
        "noticias_totales": len(documentos),
        "semilla": SEMILLA,
        "particiones": resumen,
        "distribucion_por_categoria": pd.crosstab(documentos["category"], documentos["split"]).to_dict(),
        "python": sys.version.split()[0],
        "versiones": {paquete: version(paquete) for paquete in paquetes},
    }
    (destino / "configuracion.json").write_text(
        json.dumps(configuracion, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )


def vectorizar_corrida(corrida, trabajo, sello):
    if Path(corrida).name != corrida or corrida in {"", ".", ".."}:
        raise ValueError("Indica el nombre de una corrida dentro de N-gramas.")

    referencia = None
    destinos = []
    for modo in VARIANTES.values():
        carpeta = CARPETAS_NORMALIZACION[modo]
        origen = RESULTADOS / carpeta / "N-gramas" / corrida
        documentos = pd.read_csv(origen / "labeled_documents.tsv", sep="\t", dtype=str, keep_default_na=False)
        if set(documentos["split"]) != set(SPLITS) or not documentos["document_id"].is_unique:
            raise ValueError(f"Las particiones deben estar completas y los IDs deben ser únicos: {origen}")
        if referencia is not None and not documentos.equals(referencia):
            raise ValueError(f"Las variantes deben compartir los documentos y sus particiones: {origen}")
        referencia = documentos

        print(f"Vectorizando {carpeta}: 7 combinaciones × 3 representaciones.")
        guardar_representaciones(origen, trabajo / carpeta, CARPETAS_REPRESENTACION, documentos)
        for representacion in CARPETAS_REPRESENTACION.values():
            destino = RESULTADOS / carpeta / representacion / sello
            destino.parent.mkdir(parents=True, exist_ok=True)
            destinos.append((trabajo / carpeta / representacion, destino))

    for origen, destino in destinos:
        origen.rename(destino)
    return [destino for _, destino in destinos]


def ejecutar(args):
    RESULTADOS.mkdir(exist_ok=True)
    sello = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    with TemporaryDirectory(prefix=".procesando-", dir=RESULTADOS) as temporal:
        trabajo = Path(temporal)
        if args.vectorizar_ngramas is not None:
            return vectorizar_corrida(args.vectorizar_ngramas, trabajo, sello)
        corpus = cargar_corpus(args.corpus)
        print(f"Corpus de entrada: {len(corpus)} noticias.")
        if args.recolectar:
            corpus = ampliar_corpus(corpus, trabajo / "rss.db", args.dias)
        documentos = preparar_documentos(corpus)
        salida = trabajo / "salida"
        salida.mkdir()
        if args.preparar_ngramas:
            corpus.to_csv(salida / "news_raw_corpus.tsv", sep="\t", index=False)
            documentos.to_csv(salida / "labeled_documents.tsv", sep="\t", index=False)
            guardar_ngramas(documentos, salida)
            destinos = []
            for variante, modo in VARIANTES.items():
                carpeta = salida / variante
                for nombre in ("news_raw_corpus.tsv", "labeled_documents.tsv", "combinaciones.json"):
                    copyfile(salida / nombre, carpeta / nombre)
                destino = RESULTADOS / CARPETAS_NORMALIZACION[modo] / "N-gramas" / sello
                destino.parent.mkdir(parents=True, exist_ok=True)
                destinos.append((carpeta, destino))
            for carpeta, destino in destinos:
                carpeta.rename(destino)
            return [destino for _, destino in destinos]

        destino = (
            RESULTADOS / CARPETAS_NORMALIZACION[args.normalizacion]
            / CARPETAS_REPRESENTACION[args.representacion]
            / f"{sello}_{args.normalizacion}_{args.ngramas}"
        )
        pipeline = crear_pipeline(args.normalizacion, args.representacion, args.ngramas)
        textos_train = documentos.loc[documentos["split"].eq("train"), "text"]
        print("Normalizando train y ajustando el vocabulario...")
        pipeline.fit(textos_train)
        guardar_resultados(corpus, documentos, pipeline, salida, args)
        destino.parent.mkdir(parents=True, exist_ok=True)
        salida.rename(destino)
    return [destino]


if __name__ == "__main__":
    try:
        for destino in ejecutar(argumentos()):
            print(f"Resultados: {destino}")
    except (OSError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from None
