"""Exporta n-gramas por documento sin ajustar un vocabulario."""

import json
from itertools import combinations

from normalizacion import NormalizadorNoticias, tokenizar


VARIANTES = {
    "tokenizado": "minima",
    "sin_stopwords": "sin_stopwords",
    "lematizado": "lemas",
}
ORDENES = {1: "unigramas", 2: "bigramas", 3: "trigramas"}
COMBINACIONES = {
    "_".join(ORDENES[orden] for orden in ordenes): [ORDENES[orden] for orden in ordenes]
    for cantidad in range(1, len(ORDENES) + 1)
    for ordenes in combinations(ORDENES, cantidad)
}


def guardar_ngramas(documentos, destino):
    (destino / "combinaciones.json").write_text(
        json.dumps(COMBINACIONES, indent=2) + "\n", encoding="utf-8",
    )
    for carpeta, modo in VARIANTES.items():
        print(f"Generando n-gramas: {carpeta}.")
        ruta = destino / carpeta
        ruta.mkdir()
        textos = NormalizadorNoticias(modo=modo).transform(documentos["text"])
        filas = documentos[["document_id", "split"]].itertuples(index=False, name=None)
        with (ruta / "ngramas.jsonl").open("w", encoding="utf-8") as archivo:
            for (document_id, split), texto in zip(filas, textos, strict=True):
                tokens = tokenizar(texto)
                registro = {"document_id": document_id, "split": split}
                for orden, nombre in ORDENES.items():
                    registro[nombre] = [
                        " ".join(tokens[inicio:inicio + orden])
                        for inicio in range(len(tokens) - orden + 1)
                    ]
                archivo.write(json.dumps(registro, ensure_ascii=False) + "\n")
