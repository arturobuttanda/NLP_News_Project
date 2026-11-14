"""Carga el corpus de Fase 1 y permite ampliarlo con su recolector RSS."""

import sys
from collections import defaultdict
from contextlib import closing
from datetime import timedelta
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pandas as pd


SCRAPPER_DIR = Path(__file__).resolve().parents[2] / "fase_1_recuperacion" / "scraper"
CORPUS_ORIGINAL = Path(__file__).resolve().parents[2] / "data" / "raw" / "news_raw_corpus_4645.tsv"
COLUMNAS = ["id", "title", "description", "source", "category", "date", "url", "retrieved_at"]


def cargar_corpus(ruta=CORPUS_ORIGINAL):
    """Lee un corpus conservando IDs y metadatos como texto."""
    corpus = pd.read_csv(ruta, sep="\t", dtype=str, keep_default_na=False)
    if corpus.empty or not set(COLUMNAS).issubset(corpus.columns):
        raise ValueError("El corpus está vacío o le faltan columnas.")
    if not corpus["id"].is_unique or corpus["id"].str.strip().eq("").any():
        raise ValueError("El corpus debe tener IDs únicos y no vacíos.")
    return corpus[COLUMNAS]


def ampliar_corpus(corpus, ruta_bd, dias):
    """Recolecta en una BD temporal y agrega las noticias únicas por URL y título."""
    spec = spec_from_file_location("scrapper_fase1", SCRAPPER_DIR / "scraper.py")
    scraper = module_from_spec(spec)
    guardar_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(scraper)
    finally:
        sys.dont_write_bytecode = guardar_bytecode
    scraper.DB_PATH = str(ruta_bd)
    scraper.preparar_bd()

    limite = scraper.ahora_utc() - timedelta(days=dias)
    estado = defaultdict(int)
    estado["por_categoria"] = defaultdict(int)
    with closing(scraper.conectar()) as conn:
        for categoria, feeds in scraper.FEEDS.items():
            for fuente, url in feeds:
                try:
                    with conn:
                        scraper.procesar_feed(conn, categoria, fuente, url, limite, False, estado)
                except Exception as error:
                    estado["errores"] += 1
                    print(f"[ERROR] {fuente}: {error}")
        filas = conn.execute("SELECT * FROM articulos ORDER BY num").fetchall()

    urls = {scraper.canonicalizar_url(url) for url in corpus["url"]}
    titulos = {scraper.clave_titulo(titulo) for titulo in corpus["title"]}
    ids = set(corpus["id"])
    nuevas = []
    for fila in filas:
        if fila["url"] in urls or fila["title_key"] in titulos:
            continue
        # Un ID derivado de la URL se mantiene estable entre recolecciones.
        nuevo_id = f"RSS-{fila['url_hash']}"
        if nuevo_id in ids:
            raise ValueError(f"El ID RSS coincide con un ID existente: {nuevo_id}")
        nuevas.append({
            columna: nuevo_id if columna == "id" else fila[columna]
            for columna in COLUMNAS
        })

    print(f"RSS: {estado['feeds_ok']} feeds leídos, {estado['errores']} con error, "
          f"{estado['robots']} bloqueados, {len(nuevas)} noticias adicionales.")
    if estado["feeds_ok"] == 0:
        raise ValueError("No se pudo leer ningún feed RSS.")
    return pd.concat([corpus, pd.DataFrame(nuevas, columns=COLUMNAS)], ignore_index=True)
