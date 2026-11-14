"""Vectoriza las ocurrencias de n-gramas exportadas por el pipeline."""

import json
from importlib.metadata import version
from shutil import copyfile

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from ngramas import COMBINACIONES


SPLITS = ("train", "dev", "test")


def crear_vectorizador(representacion, **opciones):
    if representacion == "tfidf":
        return TfidfVectorizer(**opciones)
    return CountVectorizer(binary=representacion == "binaria", **opciones)


def ocurrencias_preparadas(ocurrencias):
    """Entrega al vectorizador los n-gramas completos, incluidos sus espacios."""
    return ocurrencias


def cargar_ngramas(origen, documentos):
    combinaciones = json.loads((origen / "combinaciones.json").read_text(encoding="utf-8"))
    if combinaciones != COMBINACIONES:
        raise ValueError(f"El catálogo de combinaciones no coincide: {origen}")

    registros = {split: [] for split in SPLITS}
    claves = documentos[["document_id", "split"]].itertuples(index=False, name=None)
    with (origen / "ngramas.jsonl").open(encoding="utf-8") as archivo:
        for linea, (document_id, split) in zip(archivo, claves, strict=True):
            registro = json.loads(linea)
            if (registro["document_id"], registro["split"]) != (document_id, split):
                raise ValueError(f"El orden de los n-gramas no coincide con los documentos: {origen}")
            registros[split].append(registro)
    return registros


def guardar_representaciones(origen, trabajo, representaciones, documentos):
    registros = cargar_ngramas(origen, documentos)
    for carpeta in representaciones.values():
        destino = trabajo / carpeta
        destino.mkdir(parents=True)
        copyfile(origen / "labeled_documents.tsv", destino / "labeled_documents.tsv")

    for combinacion, ordenes in COMBINACIONES.items():
        textos = {
            split: [[termino for orden in ordenes for termino in fila[orden]] for fila in filas]
            for split, filas in registros.items()
        }
        for representacion, carpeta in representaciones.items():
            destino = trabajo / carpeta / combinacion
            destino.mkdir()
            vectorizador = crear_vectorizador(representacion, analyzer=ocurrencias_preparadas)
            vectorizador.fit(textos["train"])
            pd.DataFrame({"term": vectorizador.get_feature_names_out()}).to_csv(
                destino / "vocabulario.tsv", sep="\t", index=False,
            )
            particiones = {}
            for split, filas in textos.items():
                matriz = vectorizador.transform(filas).tocsr()
                save_npz(destino / f"X_{split}.npz", matriz)
                particiones[split] = {
                    "filas": matriz.shape[0],
                    "columnas": matriz.shape[1],
                    "vectores_cero": int(np.count_nonzero(matriz.getnnz(axis=1) == 0)),
                }
            joblib.dump(vectorizador, destino / "vectorizador.joblib")
            configuracion = {
                "origen_ngramas": str(origen.resolve()),
                "representacion": representacion,
                "combinacion": combinacion,
                "ordenes": ordenes,
                "particion_de_ajuste": "train",
                "particiones": particiones,
                "versiones": {paquete: version(paquete) for paquete in ("scikit-learn", "scipy", "numpy", "joblib")},
            }
            (destino / "configuracion.json").write_text(
                json.dumps(configuracion, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
            )
