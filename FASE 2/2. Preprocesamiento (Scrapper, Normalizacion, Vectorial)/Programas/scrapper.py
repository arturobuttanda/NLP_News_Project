"""Carga el corpus de Fase 1 y permite ampliarlo con su recolector RSS."""

import sys
from collections import defaultdict
from contextlib import closing
from datetime import timedelta
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pandas as pd


SCRAPPER_DIR = Path(__file__).resolve().parents[3] / "FASE 1" / "1. Scrapper"
CORPUS_ORIGINAL = Path(__file__).resolve().parent / "Corpus" / "news_raw_corpus.tsv"
COLUMNAS = ["id", "title", "description", "source", "category", "date", "url", "retrieved_at"]


def cargar_corpus(ruta=CORPUS_ORIGINAL):
    """Carga el TSV y valida sus columnas e identificadores."""
    # Se fuerza texto para conservar exactamente los metadatos.
    corpus = pd.read_csv(ruta, sep="\t", dtype=str, keep_default_na=False)
    if corpus.empty or not set(COLUMNAS).issubset(corpus.columns):
        raise ValueError("El corpus está vacío o le faltan columnas.")
    if not corpus["id"].is_unique or corpus["id"].str.strip().eq("").any():
        raise ValueError("El corpus debe tener IDs únicos y no vacíos.")
    return corpus[COLUMNAS]


def ampliar_corpus(corpus, ruta_bd, dias):
    """Recolecta RSS y agrega noticias no repetidas al corpus."""
    # Se importa el scraper de Fase 1 sin crear archivos bytecode.
    spec = spec_from_file_location("scrapper_fase1", SCRAPPER_DIR / "Scrapper.py")
    scraper = module_from_spec(spec)
    guardar_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(scraper)
    finally:
        sys.dont_write_bytecode = guardar_bytecode
    scraper.DB_PATH = str(ruta_bd)
    scraper.preparar_bd()

    # Se calcula la fecha mínima aceptada para las noticias.
    limite = scraper.ahora_utc() - timedelta(days=dias)
    # Se acumulan resultados y errores de los feeds procesados.
    estado = defaultdict(int)
    estado["por_categoria"] = defaultdict(int)
    with closing(scraper.conectar()) as conn:
        # Se recorren todas las categorías y sus fuentes RSS.
        for categoria, feeds in scraper.FEEDS.items():
            # Cada fuente se procesa dentro de una transacción independiente.
            for fuente, url in feeds:
                try:
                    with conn:
                        scraper.procesar_feed(conn, categoria, fuente, url, limite, False, estado)
                except Exception as error:
                    estado["errores"] += 1
                    print(f"[ERROR] {fuente}: {error}")
        filas = conn.execute("SELECT * FROM articulos ORDER BY num").fetchall()

    # Se crean índices para descartar noticias ya existentes.
    urls = {scraper.canonicalizar_url(url) for url in corpus["url"]}
    titulos = {scraper.clave_titulo(titulo) for titulo in corpus["title"]}
    ids = set(corpus["id"])
    # Aquí se almacenan únicamente las noticias nuevas.
    nuevas = []
    # Se revisan las filas obtenidas desde la base temporal.
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
