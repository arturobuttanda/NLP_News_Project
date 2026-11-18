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
    """Crea el vectorizador solicitado para conteos o TF-IDF."""
    if representacion == "tfidf":
        return TfidfVectorizer(**opciones)
    return CountVectorizer(binary=representacion == "binaria", **opciones)


def ocurrencias_preparadas(ocurrencias):
    """Entrega al vectorizador los n-gramas completos, incluidos sus espacios."""
    # El analizador recibe directamente las ocurrencias ya construidas.
    return ocurrencias


def cargar_ngramas(origen, documentos):
    """Carga n-gramas y verifica su orden respecto a los documentos."""
    # Se valida que el catálogo de combinaciones sea el esperado.
    combinaciones = json.loads((origen / "combinaciones.json").read_text(encoding="utf-8"))
    if combinaciones != COMBINACIONES:
        raise ValueError(f"El catálogo de combinaciones no coincide: {origen}")

    # Se separan los registros por train, dev y test.
    registros = {split: [] for split in SPLITS}
    claves = documentos[["document_id", "split"]].itertuples(index=False, name=None)
    with (origen / "ngramas.jsonl").open(encoding="utf-8") as archivo:
        # Cada registro debe corresponder al documento de la misma posición.
        for linea, (document_id, split) in zip(archivo, claves, strict=True):
            registro = json.loads(linea)
            if (registro["document_id"], registro["split"]) != (document_id, split):
                raise ValueError(f"El orden de los n-gramas no coincide con los documentos: {origen}")
            registros[split].append(registro)
    return registros


def guardar_representaciones(origen, trabajo, representaciones, documentos):
    """Genera matrices y archivos para todas las representaciones."""
    # Se cargan y agrupan los n-gramas de entrada.
    registros = cargar_ngramas(origen, documentos)
    # Se crea una carpeta base por tipo de representación.
    for carpeta in representaciones.values():
        destino = trabajo / carpeta
        destino.mkdir(parents=True, exist_ok=True)
        copyfile(origen / "labeled_documents.tsv", destino / "labeled_documents.tsv")

    # Se procesa cada combinación de órdenes de n-gramas.
    for combinacion, ordenes in COMBINACIONES.items():
        # Se concatenan las ocurrencias seleccionadas por partición.
        textos = {
            split: [[termino for orden in ordenes for termino in fila[orden]] for fila in filas]
            for split, filas in registros.items()
        }
        # Se ajusta un vectorizador independiente para cada representación.
        for representacion, carpeta in representaciones.items():
            destino = trabajo / carpeta / combinacion
            destino.mkdir(parents=True, exist_ok=True)
            vectorizador = crear_vectorizador(representacion, analyzer=ocurrencias_preparadas)
            vectorizador.fit(textos["train"])
            pd.DataFrame({"term": vectorizador.get_feature_names_out()}).to_csv(
                destino / "vocabulario.tsv", sep="\t", index=False,
            )
            # Se registran dimensiones y vectores vacíos por partición.
            particiones = {}
            # Se transforman train, dev y test con el vocabulario de train.
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
