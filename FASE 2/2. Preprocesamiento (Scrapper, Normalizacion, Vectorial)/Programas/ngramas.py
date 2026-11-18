"""Exporta n-gramas por documento sin ajustar un vocabulario."""

import json
from itertools import combinations

from normalizacion import NormalizadorNoticias, tokenizar


VARIANTES = {
    "1. Ind Tokenizado": "minima",
    "1.1 Comb Stop words": "combinado_stopwords",
    "1.2 Comb Lematizado": "combinado_lemas",
    "1.3 Tokenizado Lematizado": "tokenizado_lemas",
    "1.4 Stop words Lematizado": "stopwords_lemas",
    "2. Ind Stop words": "sin_stopwords",
    "3. Ind Lematizado": "lemas",
}
ORDENES = {1: "unigramas", 2: "bigramas", 3: "trigramas"}
COMBINACIONES = {
    "_".join(ORDENES[orden] for orden in ordenes): [ORDENES[orden] for orden in ordenes]
    for cantidad in range(1, len(ORDENES) + 1)
    for ordenes in combinations(ORDENES, cantidad)
}


def guardar_ngramas(documentos, destino, modo):
    """Genera y guarda los n-gramas de una variante de normalización."""
    # Se guarda el catálogo para reconstruir las combinaciones posteriores.
    (destino / "combinaciones.json").write_text(
        json.dumps(COMBINACIONES, indent=2) + "\n", encoding="utf-8",
    )
    print(f"Generando n-gramas: {modo}.")
    # Se transforma el texto antes de extraer sus tokens.
    textos = NormalizadorNoticias(modo=modo).transform(documentos["text"])
    # Se conserva la relación entre cada documento y su partición.
    filas = documentos[["document_id", "split"]].itertuples(index=False, name=None)
    with (destino / "ngramas.jsonl").open("w", encoding="utf-8") as archivo:
        # Cada línea del archivo representa un documento.
        for (document_id, split), texto in zip(filas, textos, strict=True):
            tokens = tokenizar(texto)
            registro = {"document_id": document_id, "split": split}
            # Se generan unigramas, bigramas y trigramas.
            for orden, nombre in ORDENES.items():
                registro[nombre] = [
                    " ".join(tokens[inicio:inicio + orden])
                    for inicio in range(len(tokens) - orden + 1)
                ]
            archivo.write(json.dumps(registro, ensure_ascii=False) + "\n")
