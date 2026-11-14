
import calendar
import csv
import hashlib
import html
import os
import re
import sqlite3
import sys
import time
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from urllib.robotparser import RobotFileParser

import feedparser
import requests
from bs4 import BeautifulSoup
from dateutil import parser as date_parser

warnings.filterwarnings("ignore", category=UserWarning, module="bs4")

# CONFIGURACION
#rutas de la bd y de los archivos de salida, todo junto al script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
DATA_RAW_DIR = os.path.join(ROOT_DIR, "data", "raw")
DB_PATH = os.path.join(DATA_RAW_DIR, "noticias.db") if os.path.exists(DATA_RAW_DIR) else os.path.join(BASE_DIR, "noticias.db")
RAW_CORPUS_PATH = os.path.join(DATA_RAW_DIR, "news_raw_corpus_4645.tsv") if os.path.exists(DATA_RAW_DIR) else os.path.join(BASE_DIR, "news_raw_corpus.tsv")
VALIDACION_PATH = os.path.join(BASE_DIR, "feeds_validacion.tsv")

# Cambia el correo: es buena practica que el sitio pueda contactarte.
USER_AGENT = "NLP-Course-Bot/1.0 (academic project; contact: tu_correo@ejemplo.com)"
#cabeceras que se mandan en cada peticion http (identifican al bot)
HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "application/rss+xml, application/atom+xml, application/xml;q=0.9, text/xml;q=0.8, */*;q=0.5",
}

#parametros de red; no se limita la cantidad ni la antiguedad de las entradas
DELAY_POR_DOMINIO = 1.5   # segundos minimos entre solicitudes al mismo dominio
TIMEOUT = 20
MAX_REINTENTOS = 3

# (nombre_del_medio, url_del_feed)
# Cada feed pertenece a UNA categoria. Si una nota aparece en varios feeds,
# se queda con la categoria del primero que la trae (orden de este dict).
# Los feeds cambian o desaparecen: se validan durante la recoleccion.
#diccionario categoria -> lista de (medio, url del rss)
FEEDS = {
    "politica": [
        ("BBC News", "https://feeds.bbci.co.uk/news/politics/rss.xml"),
        ("The Guardian", "https://www.theguardian.com/politics/rss"),
        ("The Guardian", "https://www.theguardian.com/us-news/us-politics/rss"),
        ("NPR", "https://feeds.npr.org/1014/rss.xml"),
        ("Politico", "https://rss.politico.com/politics-news.xml"),
        ("CBS News", "https://www.cbsnews.com/latest/rss/politics"),
        ("ABC News US", "https://feeds.abcnews.com/abcnews/politicsheadlines"),
        ("Sky News", "https://feeds.skynews.com/feeds/rss/politics.xml"),
        ("The Hindu", "https://www.thehindu.com/news/national/feeder/default.rss"),
    ],
    "economia": [
        ("BBC News", "https://feeds.bbci.co.uk/news/business/rss.xml"),
        ("The Guardian", "https://www.theguardian.com/business/rss"),
        ("NPR", "https://feeds.npr.org/1006/rss.xml"),
        ("CNBC", "https://www.cnbc.com/id/10001147/device/rss/rss.html"),
        ("CNBC", "https://www.cnbc.com/id/20910258/device/rss/rss.html"),
        ("CNBC", "https://www.cnbc.com/id/10000664/device/rss/rss.html"),
        ("Sky News", "https://feeds.skynews.com/feeds/rss/business.xml"),
        ("The Hindu", "https://www.thehindu.com/business/feeder/default.rss"),
    ],
    "tecnologia_ia": [
        ("Ars Technica", "https://feeds.arstechnica.com/arstechnica/index"),
        ("The Verge", "https://www.theverge.com/rss/index.xml"),
        ("Wired", "https://www.wired.com/feed/rss"),
        ("TechCrunch", "https://techcrunch.com/feed/"),
        ("MIT Technology Review", "https://www.technologyreview.com/feed/"),
        ("Engadget", "https://www.engadget.com/rss.xml"),
        ("The Guardian", "https://www.theguardian.com/technology/rss"),
        ("The Guardian", "https://www.theguardian.com/technology/artificialintelligenceai/rss"),
        ("BBC News", "https://feeds.bbci.co.uk/news/technology/rss.xml"),
        ("Sky News", "https://feeds.skynews.com/feeds/rss/technology.xml"),
        ("CNBC", "https://www.cnbc.com/id/19854910/device/rss/rss.html"),
    ],
    "ciencia": [
        ("Nature", "https://www.nature.com/nature.rss"),
        ("Nature Communications", "https://www.nature.com/ncomms.rss"),
        ("Scientific Reports", "https://www.nature.com/srep.rss"),
        ("ScienceDaily", "https://www.sciencedaily.com/rss/top/science.xml"),
        ("Phys.org", "https://phys.org/rss-feed/"),
        ("NASA", "https://www.nasa.gov/feed/"),
        ("Scientific American", "https://www.scientificamerican.com/platform/syndication/rss/"),
        ("Science News", "https://www.sciencenews.org/feed"),
        ("Live Science", "https://www.livescience.com/feeds/all"),
        ("Space.com", "https://www.space.com/feeds/all"),
        ("NPR", "https://feeds.npr.org/1007/rss.xml"),
    ],
    "salud": [
        ("BBC News", "https://feeds.bbci.co.uk/news/health/rss.xml"),
        ("The Guardian", "https://www.theguardian.com/society/health/rss"),
        ("NPR", "https://feeds.npr.org/1128/rss.xml"),
        ("STAT News", "https://www.statnews.com/feed/"),
        ("ScienceDaily", "https://www.sciencedaily.com/rss/health_medicine.xml"),
        ("WHO", "https://www.who.int/rss-feeds/news-english.xml"),
        ("Fierce Healthcare", "https://www.fiercehealthcare.com/rss/xml"),
    ],
    "medio_ambiente": [
        ("The Guardian", "https://www.theguardian.com/environment/rss"),
        ("The Guardian", "https://www.theguardian.com/environment/climate-crisis/rss"),
        ("Carbon Brief", "https://www.carbonbrief.org/feed/"),
        ("Mongabay", "https://news.mongabay.com/feed/"),
        ("Inside Climate News", "https://insideclimatenews.org/feed/"),
        ("Grist", "https://grist.org/feed/"),
        ("Yale Environment 360", "https://e360.yale.edu/feed.xml"),
        ("Climate Home News", "https://www.climatechangenews.com/feed/"),
        ("ScienceDaily", "https://www.sciencedaily.com/rss/earth_climate.xml"),
        ("NPR", "https://feeds.npr.org/1025/rss.xml"),
    ],
    "deportes": [
        ("BBC Sport", "https://feeds.bbci.co.uk/sport/rss.xml"),
        ("The Guardian", "https://www.theguardian.com/uk/sport/rss"),
        ("ESPN", "https://www.espn.com/espn/rss/news"),
        ("ESPN", "https://www.espn.com/espn/rss/nfl/news"),
        ("ESPN", "https://www.espn.com/espn/rss/nba/news"),
        ("ESPN", "https://www.espn.com/espn/rss/mlb/news"),
        ("ESPN", "https://www.espn.com/espn/rss/soccer/news"),
        ("CBS Sports", "https://www.cbssports.com/rss/headlines/"),
        ("Sky Sports", "https://www.skysports.com/rss/12040"),
    ],
    "mundo_conflictos": [
        ("BBC News", "https://feeds.bbci.co.uk/news/world/rss.xml"),
        ("BBC News", "https://feeds.bbci.co.uk/news/world/middle_east/rss.xml"),
        ("BBC News", "https://feeds.bbci.co.uk/news/world/africa/rss.xml"),
        ("BBC News", "https://feeds.bbci.co.uk/news/world/asia/rss.xml"),
        ("The Guardian", "https://www.theguardian.com/world/rss"),
        ("The Guardian", "https://www.theguardian.com/world/ukraine/rss"),
        ("Al Jazeera", "https://www.aljazeera.com/xml/rss/all.xml"),
        ("NPR", "https://feeds.npr.org/1004/rss.xml"),
        ("Sky News", "https://feeds.skynews.com/feeds/rss/world.xml"),
        ("DW", "https://rss.dw.com/rdf/rss-en-world"),
        ("France 24", "https://www.france24.com/en/rss"),
    ],
    "cultura_espectaculos": [
        ("BBC News", "https://feeds.bbci.co.uk/news/entertainment_and_arts/rss.xml"),
        ("The Guardian", "https://www.theguardian.com/culture/rss"),
        ("The Guardian", "https://www.theguardian.com/film/rss"),
        ("The Guardian", "https://www.theguardian.com/music/rss"),
        ("Variety", "https://variety.com/feed/"),
        ("Deadline", "https://deadline.com/feed/"),
        ("The Hollywood Reporter", "https://www.hollywoodreporter.com/feed/"),
        ("NPR", "https://feeds.npr.org/1008/rss.xml"),
        ("Pitchfork", "https://pitchfork.com/feed/feed-news/rss"),
        ("Rolling Stone", "https://www.rollingstone.com/feed/"),
        ("The Hindu", "https://www.thehindu.com/entertainment/feeder/default.rss"),
    ],
    "sociedad": [  # migracion, educacion, justicia, vida social
        ("The Guardian", "https://www.theguardian.com/society/rss"),
        ("The Guardian", "https://www.theguardian.com/education/rss"),
        ("The Guardian", "https://www.theguardian.com/law/rss"),
        ("The Guardian", "https://www.theguardian.com/uk/ukcrime/rss"),
        ("BBC News", "https://feeds.bbci.co.uk/news/education/rss.xml"),
        ("BBC News", "https://feeds.bbci.co.uk/news/uk/rss.xml"),
        ("BBC News", "https://feeds.bbci.co.uk/news/world/us_and_canada/rss.xml"),
        ("NPR", "https://feeds.npr.org/1013/rss.xml"),
        ("Sky News", "https://feeds.skynews.com/feeds/rss/uk.xml"),
        ("Sky News", "https://feeds.skynews.com/feeds/rss/us.xml"),
        ("RTE", "https://www.rte.ie/feeds/rss/?index=/news/"),
    ],
}

# Abreviaturas de zona horaria que dateutil NO conoce por defecto
# (sin esto asume UTC y corrompe la hora). Solo se usa como respaldo:
# la via principal es entry.published_parsed de feedparser.
# Nota: IST es ambiguo (India / Irlanda / Israel); se asume India (+5:30).
#abreviatura de zona horaria -> segundos de diferencia con UTC
TZINFOS = {
    "UTC": 0, "GMT": 0, "Z": 0, "UT": 0,
    "EST": -5 * 3600, "EDT": -4 * 3600, "CST": -6 * 3600, "CDT": -5 * 3600,
    "MST": -7 * 3600, "MDT": -6 * 3600, "PST": -8 * 3600, "PDT": -7 * 3600,
    "AKST": -9 * 3600, "AKDT": -8 * 3600, "HST": -10 * 3600,
    "BST": 1 * 3600, "WET": 0, "WEST": 1 * 3600, "CET": 1 * 3600, "CEST": 2 * 3600,
    "EET": 2 * 3600, "EEST": 3 * 3600, "IST": 19800, "PKT": 18000,
    "SGT": 8 * 3600, "HKT": 8 * 3600, "AWST": 8 * 3600, "JST": 9 * 3600, "KST": 9 * 3600,
    "ACST": 34200, "ACDT": 37800, "AEST": 10 * 3600, "AEDT": 11 * 3600,
    "NZST": 12 * 3600, "NZDT": 13 * 3600, "SAST": 2 * 3600, "WAT": 1 * 3600, "EAT": 3 * 3600,
}

#parametros de url que solo sirven para rastreo; se eliminan al canonicalizar
TRACKING_PARAMS = {
    "at_medium", "at_campaign", "at_custom1", "at_custom2", "at_custom3", "at_custom4",
    "ocid", "cmp", "fbclid", "gclid", "taid", "guccounter", "ito", "cid", "ftag",
    "mc_cid", "mc_eid", "smid", "partner", "rss", "xrs",
}

# UTILIDADES

def pausar():
    input("\nPresione Enter para volver al menu...")


def mostrar_seccion(titulo):
    titulo = titulo.upper()
    print(titulo)
    print("*" * len(titulo))


def ahora_utc():
    return datetime.now(timezone.utc).replace(microsecond=0)


def canonicalizar_url(url):
    """Normaliza una URL para detectar duplicados entre corridas y feeds."""
    #separa la url en sus partes
    p = urlparse(url.strip())
    #dominio en minusculas y sin 'www.' para que una misma nota no tenga dos urls
    host = p.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    #en bbc se borran todos los parametros
    if host.endswith(("bbc.co.uk", "bbc.com")):
        query = []  # BBC solo usa parametros de tracking
    else:
        #en el resto de sitios solo se quitan los parametros de tracking
        query = [
            (k, v) for k, v in parse_qsl(p.query, keep_blank_values=True)
            if not k.lower().startswith("utm_") and k.lower() not in TRACKING_PARAMS
        ]
    #quita la barra final; se fuerza https y se descarta el fragmento (#...)
    path = p.path.rstrip("/") or "/"
    return urlunparse(("https", host, path, "", urlencode(query), ""))


def hash_url(url_canonica):
    #sha1 de la url canonica: sirve como llave unica en la bd
    return hashlib.sha1(url_canonica.encode("utf-8")).hexdigest()


def clave_titulo(titulo):
    """Titulo en minusculas, solo alfanumerico: detecta la misma nota en varios feeds/medios."""
    #deja solo letras y numeros en minuscula para comparar titulos sin importar signos
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", titulo.lower())).strip()


def limpiar_texto(raw):
    """Quita HTML, decodifica entidades y colapsa espacios/tabs/saltos/control chars."""
    if not raw:
        return ""
    #aqui beautifulsoup solo quita las etiquetas html del resumen que trae el feed
    texto = BeautifulSoup(raw, "html.parser").get_text(" ")
    #convierte entidades como &amp; a su caracter
    texto = html.unescape(texto)
    #cambia caracteres de control y espacios raros por un espacio
    texto = re.sub(r"[\x00-\x1f\x7f\u200b\ufeff\xa0]+", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


def extraer_fecha(entry):
    """
    Devuelve (fecha_iso_utc | None, fecha_cruda).
    1) published/updated_parsed de feedparser (ya convertido a UTC).
    2) Respaldo: dateutil con diccionario de zonas horarias.
    """
    #fecha tal cual viene en el feed (se guarda como respaldo)
    cruda = entry.get("published") or entry.get("updated") or entry.get("created") or ""
    dt = None
    #via principal: feedparser ya entrega la fecha parseada en UTC
    st = entry.get("published_parsed") or entry.get("updated_parsed")
    if st:
        try:
            #convierte la tupla a datetime con zona UTC
            dt = datetime.fromtimestamp(calendar.timegm(st), tz=timezone.utc)
        except (ValueError, OverflowError, OSError):
            dt = None
    if dt is None and cruda:
        try:
            #respaldo: parsea el texto de la fecha usando el diccionario de zonas
            d = date_parser.parse(cruda, tzinfos=TZINFOS)
            dt = d.replace(tzinfo=timezone.utc) if d.tzinfo is None else d.astimezone(timezone.utc)
        except (ValueError, OverflowError, TypeError):
            dt = None
    #devuelve la fecha en iso sin microsegundos, o None si no se pudo
    return (dt.replace(microsecond=0).isoformat() if dt else None), cruda


# RED: robots.txt, pausas por dominio y reintentos
_robots = {}
_ultima_peticion = {}


def robots_permite(url):
    """(permitido, crawl_delay). 401/403 en robots.txt = prohibido; 404 = permitido."""
    #robots.txt se pide una sola vez por dominio y queda en cache (_robots)
    p = urlparse(url)
    base = f"{p.scheme}://{p.netloc}"
    if base not in _robots:
        rp = RobotFileParser()
        try:
            r = requests.get(base + "/robots.txt", headers=HEADERS, timeout=TIMEOUT)
            #401/403 se interpreta como prohibido
            if r.status_code in (401, 403):
                rp.disallow_all = True
            #si existe, se parsea para consultar sus reglas
            elif r.status_code == 200:
                rp.parse(r.text.splitlines())
                rp.modified()
            else:
                rp.allow_all = True
        #si no se puede leer robots.txt se asume permitido
        except requests.RequestException:
            rp.allow_all = True
        _robots[base] = rp
    rp = _robots[base]
    #devuelve (permitido?, crawl-delay que pida el sitio)
    return rp.can_fetch(USER_AGENT, url), rp.crawl_delay(USER_AGENT)


def esperar_turno(dominio, delay):
    #instante de la ultima peticion a este dominio
    previo = _ultima_peticion.get(dominio)
    if previo is not None:
        #si no paso el delay minimo, duerme lo que falta
        falta = delay - (time.monotonic() - previo)
        if falta > 0:
            time.sleep(falta)
    _ultima_peticion[dominio] = time.monotonic()


def descargar(url, delay, etag=None, last_mod=None):
    """GET con reintentos. Devuelve (response | None, mensaje_error | None)."""
    #si hay etag/last-modified del cache se mandan: el servidor responde 304 si no hubo cambios
    headers = dict(HEADERS)
    if etag:
        headers["If-None-Match"] = etag
    if last_mod:
        headers["If-Modified-Since"] = last_mod
    dominio = urlparse(url).netloc
    error = None
    #hasta MAX_REINTENTOS intentos
    for intento in range(1, MAX_REINTENTOS + 1):
        esperar_turno(dominio, delay)
        try:
            r = requests.get(url, headers=headers, timeout=TIMEOUT)
        #error de red: espera exponencial (2, 4, 8 s) y reintenta
        except requests.RequestException as e:
            error = f"{type(e).__name__}: {e}"
            time.sleep(2 ** intento)
            continue
        #el sitio pide bajar el ritmo: respeta Retry-After o espera exponencial (maximo 60 s)
        if r.status_code in (429, 503):
            ra = r.headers.get("Retry-After", "")
            espera = int(ra) if ra.isdigit() else 5 * 2 ** (intento - 1)
            error = f"HTTP {r.status_code} (reintento {intento}/{MAX_REINTENTOS})"
            time.sleep(min(espera, 60))
            continue
        #cualquier otro codigo se devuelve y lo evalua quien llamo
        return r, None
    return None, error or "sin respuesta"


# BASE DE DATOS

def preparar_bd():
    """Si existe una BD con el esquema viejo, la respalda y crea una nueva."""
    #si hay una bd con esquema viejo (sin url_hash) se renombra como respaldo
    if os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        cols = [r[1] for r in conn.execute("PRAGMA table_info(articulos)")]
        conn.close()
        if cols and "url_hash" not in cols:
            respaldo = os.path.join(BASE_DIR, f"noticias_antigua_{int(time.time())}.db")
            os.rename(DB_PATH, respaldo)
            print(f"[!] BD con esquema anterior respaldada en: {respaldo}")

    #tablas: articulos (corpus, num autoincremental), log_errores y feed_cache (etag/last-modified)
    conn = sqlite3.connect(DB_PATH)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS articulos (
            num          INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id       TEXT UNIQUE,
            url_hash     TEXT NOT NULL UNIQUE,
            title_key    TEXT NOT NULL UNIQUE,
            title        TEXT NOT NULL,
            description  TEXT NOT NULL,
            source       TEXT NOT NULL,
            category     TEXT NOT NULL,
            date         TEXT NOT NULL,
            date_raw     TEXT,
            url          TEXT NOT NULL,
            feed_url     TEXT,
            retrieved_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS log_errores (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ts          TEXT NOT NULL,
            feed_url    TEXT,
            http_status INTEGER,
            mensaje     TEXT
        );
        CREATE TABLE IF NOT EXISTS feed_cache (
            feed_url      TEXT PRIMARY KEY,
            etag          TEXT,
            last_modified TEXT
        );
    """)
    conn.commit()
    conn.close()


def conectar():
    conn = sqlite3.connect(DB_PATH)
    #permite leer columnas por nombre: fila['title']
    conn.row_factory = sqlite3.Row
    return conn


def registrar_error(conn, feed_url, status, mensaje):
    #guarda el error con fecha; el mensaje se corta a 500 caracteres
    conn.execute(
        "INSERT INTO log_errores (ts, feed_url, http_status, mensaje) VALUES (?,?,?,?)",
        (ahora_utc().isoformat(), feed_url, status, str(mensaje)[:500]),
    )


def insertar_articulo(conn, a):
    """
    Inserta si no es duplicado. Devuelve 'nuevo', 'dup_url' o 'dup_titulo'.
    doc_id = N + num autoincremental: estable entre corridas y nunca reutilizado.
    """
    #primero se revisa duplicado por url
    if conn.execute("SELECT 1 FROM articulos WHERE url_hash=?", (a["url_hash"],)).fetchone():
        return "dup_url"
    #luego por titulo (misma nota en otro feed u otro medio)
    if conn.execute("SELECT 1 FROM articulos WHERE title_key=?", (a["title_key"],)).fetchone():
        return "dup_titulo"
    #inserta la fila; los :nombres se rellenan con el diccionario a
    cur = conn.execute(
        """INSERT INTO articulos (url_hash, title_key, title, description, source, category,
                                  date, date_raw, url, feed_url, retrieved_at)
           VALUES (:url_hash,:title_key,:title,:description,:source,:category,
                   :date,:date_raw,:url,:feed_url,:retrieved_at)""", a)
    #el doc_id (N0001...) se arma con el num que asigno sqlite
    conn.execute("UPDATE articulos SET doc_id=? WHERE num=?", (f"N{cur.lastrowid:04d}", cur.lastrowid))
    return "nuevo"


# RECOLECCION

def procesar_feed(conn, categoria, fuente, feed_url, usar_cache, res):
    #1) respeta robots.txt: si lo bloquea, el feed se omite
    permitido, crawl_delay = robots_permite(feed_url)
    if not permitido:
        res["robots"] += 1
        registrar_error(conn, feed_url, None, "Bloqueado por robots.txt (omitido)")
        print(f"  [ROBOTS] {fuente}: bloqueado, se omite -> {feed_url}")
        return
    #pausa entre peticiones: la mayor entre la configurada y el crawl-delay (tope 10 s)
    delay = min(max(DELAY_POR_DOMINIO, crawl_delay or 0), 10)

    #2) si hay cache, recupera etag/last-modified de la corrida anterior
    etag = last_mod = None
    if usar_cache:
        fila = conn.execute("SELECT etag, last_modified FROM feed_cache WHERE feed_url=?", (feed_url,)).fetchone()
        if fila:
            etag, last_mod = fila["etag"], fila["last_modified"]

    #3) descarga el feed con reintentos
    r, err = descargar(feed_url, delay, etag, last_mod)
    if r is None:
        res["errores"] += 1
        registrar_error(conn, feed_url, None, err)
        print(f"  [ERROR] {fuente}: {err}")
        return
    #304 = el feed no cambio, no hay nada que procesar
    if r.status_code == 304:
        res["sin_cambios"] += 1
        print(f"  [304]   {fuente}: sin cambios desde la ultima corrida")
        return
    if r.status_code != 200:
        res["errores"] += 1
        registrar_error(conn, feed_url, r.status_code, f"HTTP {r.status_code}")
        print(f"  [HTTP {r.status_code}] {fuente} -> {feed_url}")
        return

    #4) feedparser convierte el xml (rss/atom) en entradas; no se scrapea html de las paginas, solo se lee el feed
    feed = feedparser.parse(r.content)
    if not feed.entries:
        res["errores"] += 1
        registrar_error(conn, feed_url, 200, "Respuesta sin entradas (no es un feed valido o esta vacio)")
        print(f"  [VACIO] {fuente}: sin entradas -> {feed_url}")
        return

    #guarda etag/last-modified para la proxima corrida
    conn.execute("INSERT OR REPLACE INTO feed_cache VALUES (?,?,?)",
                 (feed_url, r.headers.get("ETag"), r.headers.get("Last-Modified")))

    nuevos = 0
    retrieved_at = ahora_utc().isoformat()
    #5) procesa cada entrada con enlace y fecha valida, sin filtros de antiguedad o longitud
    for e in feed.entries:
        link = e.get("link", "")
        if not link:
            continue
        fecha, fecha_cruda = extraer_fecha(e)
        if not fecha:
            res["sin_fecha"] += 1
            fecha = retrieved_at

        #limpia titulo y descripcion (html, entidades, espacios)
        titulo = limpiar_texto(e.get("title", ""))
        desc = limpiar_texto(e.get("summary", e.get("description", "")))

        #normaliza la url e intenta insertar; devuelve 'nuevo' o el tipo de duplicado
        url_can = canonicalizar_url(link)
        estado = insertar_articulo(conn, {
            "url_hash": hash_url(url_can),
            "title_key": clave_titulo(titulo) or f"__sin_titulo__{hash_url(url_can)}",
            "title": titulo, "description": desc, "source": fuente,
            "category": categoria, "date": fecha, "date_raw": fecha_cruda,
            "url": url_can, "feed_url": feed_url, "retrieved_at": retrieved_at,
        })
        #cuenta como nuevo o suma al contador del tipo de duplicado
        if estado == "nuevo":
            nuevos += 1
            res["nuevos"] += 1
            res["por_categoria"][categoria] += 1
        else:
            res[estado] += 1
    res["feeds_ok"] += 1
    print(f"  [OK]    {fuente}: {len(feed.entries)} entradas, {nuevos} nuevas")


def recolectar_noticias():
    mostrar_seccion("RECOLECCION DE NOTICIAS")
    preparar_bd()
    #el cache ETag puede devolver 304; ignorarlo fuerza la descarga del feed
    resp = input("Ignorar cache ETag/Last-Modified? (s/n): ").strip().lower()
    usar_cache = resp != "s"

    print("Se procesaran todas las entradas disponibles en cada feed. Esto puede tardar varios minutos.\n")

    conn = conectar()
    total_antes = conn.execute("SELECT COUNT(*) FROM articulos").fetchone()[0]
    #contadores de la corrida (nuevos, duplicados, descartados, errores)
    res = defaultdict(int)
    res["por_categoria"] = defaultdict(int)

    try:
        #recorre categoria por categoria y feed por feed
        for categoria, feeds in FEEDS.items():
            print(f"\n--- {categoria} ---")
            for fuente, url in feeds:
                try:
                    procesar_feed(conn, categoria, fuente, url, usar_cache, res)
                except Exception as ex:  # un feed roto no debe tumbar la corrida
                    res["errores"] += 1
                    registrar_error(conn, url, None, f"Excepcion: {ex}")
                    print(f"  [EXCEPCION] {fuente}: {ex}")
                #guarda despues de cada feed por si se interrumpe la corrida
                conn.commit()
    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario. Se conserva lo ya guardado.")
    conn.commit()

    #a partir de aqui solo se imprime el resumen de la corrida
    total_despues = conn.execute("SELECT COUNT(*) FROM articulos").fetchone()[0]
    print()
    mostrar_seccion("REPORTE DE LA CORRIDA")
    print(f"Feeds OK: {res['feeds_ok']} | sin cambios (304): {res['sin_cambios']} | "
          f"con error: {res['errores']} | bloqueados por robots: {res['robots']}")
    print(f"Articulos nuevos:               {res['nuevos']}")
    print(f"Duplicados por URL:             {res['dup_url']}")
    print(f"Duplicados por titulo:          {res['dup_titulo']}")
    print(f"Entradas sin fecha (usaron fecha de recoleccion): {res['sin_fecha']}")
    print("\nNuevos por categoria en esta corrida:")
    for cat in FEEDS:
        print(f"  {cat:<22} {res['por_categoria'][cat]}")
    print(f"\nTotal en la BD: {total_antes} -> {total_despues}")
    conn.close()
    pausar()


# ESTADISTICAS

def _bd_lista():
    #evita consultar si todavia no se creo la bd
    if not os.path.exists(DB_PATH):
        print("Aun no existe la base de datos. Ejecute primero la opcion 1.")
        pausar()
        return False
    return True


def estadisticas():
    mostrar_seccion("ESTADISTICAS")
    if not _bd_lista():
        return
    conn = conectar()
    #cada bloque es un GROUP BY sobre la tabla articulos
    total = conn.execute("SELECT COUNT(*) FROM articulos").fetchone()[0]
    print(f"Total: {total}")
    if total == 0:
        conn.close()
        pausar()
        return
    print("\nPor categoria:")
    for f in conn.execute("SELECT category, COUNT(*) c FROM articulos GROUP BY category ORDER BY c DESC"):
        print(f"  {f['category']:<22} {f['c']}")
    print("\nPor fuente:")
    for f in conn.execute("SELECT source, COUNT(*) c FROM articulos GROUP BY source ORDER BY c DESC"):
        print(f"  {f['source']:<26} {f['c']}")
    print("\nPor fecha de publicacion (UTC, todos los datos):")
    #substr(date,1,10) toma solo AAAA-MM-DD para contar por dia
    for f in conn.execute("SELECT substr(date,1,10) d, COUNT(*) c FROM articulos GROUP BY d ORDER BY d DESC"):
        print(f"  {f['d']}: {f['c']}")
    rango = conn.execute("SELECT MIN(date) a, MAX(date) b FROM articulos").fetchone()
    print(f"\nRango de fechas: {rango['a']}  ->  {rango['b']}")
    conn.close()
    pausar()


# VALIDAR FEEDS

def validar_feeds():
    mostrar_seccion("VALIDACION DE FEEDS (NO INSERTA ARTICULOS)")
    filas = []
    try:
        for categoria, feeds in FEEDS.items():
            print(f"\n--- {categoria} ---")
            for fuente, url in feeds:
                #estados posibles: OK, ROBOTS, ERROR, HTTP, VACIO
                estado, status, n, recientes, nuevo = "OK", "", 0, 0, ""
                #prueba cada feed: robots, descarga, parseo y conteo de entradas recientes
                permitido, crawl_delay = robots_permite(url)
                if not permitido:
                    estado = "ROBOTS"
                else:
                    r, err = descargar(url, min(max(DELAY_POR_DOMINIO, crawl_delay or 0), 10))
                    if r is None:
                        estado, status = "ERROR", err
                    elif r.status_code != 200:
                        estado, status = "HTTP", str(r.status_code)
                    else:
                        feed = feedparser.parse(r.content)
                        n = len(feed.entries)
                        #cuenta las entradas con fecha sin aplicar una ventana temporal
                        fechas = [extraer_fecha(e)[0] for e in feed.entries]
                        fechas = [f for f in fechas if f]
                        recientes = len(fechas)
                        nuevo = max(fechas)[:10] if fechas else ""
                        if n == 0:
                            estado = "VACIO"
                print(f"  [{estado:<13}] {fuente:<22} entradas={n:<3} recientes={recientes:<3} {status} {url}")
                filas.append([categoria, fuente, url, estado, status, n, recientes, nuevo])
    except KeyboardInterrupt:
        print("\nInterrumpido.")

    #guarda el resultado de cada feed en feeds_validacion.tsv
    with open(VALIDACION_PATH, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["categoria", "fuente", "url", "estado", "detalle", "entradas", "recientes", "fecha_mas_reciente"])
        w.writerows(filas)

    print()
    mostrar_seccion("RESULTADO DE VALIDACION")
    #todo lo que no quedo en OK se lista para eliminar o reemplazar
    malos = [x for x in filas if x[3] != "OK"]
    print(f"Feeds probados: {len(filas)} | OK: {len(filas) - len(malos)} | a revisar: {len(malos)}")
    for x in malos:
        print(f"  - [{x[3]}] {x[1]} ({x[0]}): {x[2]}")
    print(f"\nDetalle guardado en: {VALIDACION_PATH}")
    print("Elimine o reemplace en FEEDS los que no esten OK. NO intente evadir bloqueos.")
    pausar()


# EXPORTAR


def exportar_corpus():
    mostrar_seccion("EXPORTAR news_raw_corpus.tsv")
    if not _bd_lista():
        return
    conn = conectar()
    #lee todo el corpus guardado
    filas = conn.execute("SELECT * FROM articulos ORDER BY num").fetchall()
    conn.close()
    if not filas:
        print("La base de datos esta vacia.")
        pausar()
        return

    print("ADVERTENCIA: una vez que empiece a anotar qrels.tsv, el corpus debe quedar")
    print("congelado. Exporte la version final y no la regenere con otra seleccion.\n")
    #pide confirmacion antes de sobrescribir el tsv
    if os.path.exists(RAW_CORPUS_PATH):
        if input(f"{RAW_CORPUS_PATH} ya existe. Sobrescribir? (s/n): ").strip().lower() != "s":
            print("Exportacion cancelada.")
            pausar()
            return

    #columnas del archivo final
    columnas = ["id", "title", "description", "source", "category", "date", "url", "retrieved_at"]
    #escribe el tsv (separador tab, utf-8)
    with open(RAW_CORPUS_PATH, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(columnas)
        for r in filas:
            w.writerow([r["doc_id"], r["title"], r["description"], r["source"],
                        r["category"], r["date"], r["url"], r["retrieved_at"]])

    print(f"\nExportado: {RAW_CORPUS_PATH}")
    print(f"Filas: {len(filas)} | Columnas: {', '.join(columnas)}")

    try:  # verificacion opcional
        import pandas as pd
        df = pd.read_csv(RAW_CORPUS_PATH, sep="\t", dtype=str, keep_default_na=False)
        print(f"Verificacion pandas: {df.shape[0]} filas x {df.shape[1]} columnas")
        #cuenta campos vacios por columna
        vacios = {c: int((df[c].str.strip() == "").sum()) for c in columnas}
        vacios = {c: n for c, n in vacios.items() if n}
        print(f"Campos vacios: {vacios if vacios else 'ninguno'}")
        print(f"IDs unicos: {df['id'].is_unique}")
        print("\nDistribucion por categoria:\n" + df["category"].value_counts().to_string())
        print("\nDistribucion por fuente:\n" + df["source"].value_counts().to_string())
    except ImportError:
        print("(pandas no instalado: se omitio la verificacion)")
    pausar()


# LOG DE ERRORES

def ver_log():
    mostrar_seccion("LOG DE ERRORES (ULTIMOS 40)")
    if not _bd_lista():
        return
    conn = conectar()
    #ultimos 40 errores registrados
    filas = conn.execute("SELECT * FROM log_errores ORDER BY id DESC LIMIT 40").fetchall()
    if not filas:
        print("Sin errores registrados.")
    for f in filas:
        print(f"{f['ts']} | HTTP {f['http_status'] or '-'} | {f['feed_url']}\n    {f['mensaje']}")
    conn.close()
    pausar()


# MENU

def menu_principal():
    #opcion del menu -> funcion
    acciones = {
        "1": recolectar_noticias, "2": estadisticas, "3": exportar_corpus,
    }
    while True:
        print()
        mostrar_seccion("SCRAPER DE NOTICIAS - ETAPA 1")
        print("  1. Recolectar noticias")
        print("  2. Estadisticas (categoria, fuente y fecha)")
        print("  3. Exportar news_raw_corpus.tsv")
        print("  4. Salir")
        op = input("Seleccione una opcion (1-4): ").strip()
        if op == "4":
            print("Saliendo.")
            sys.exit(0)
        if op in acciones:
            #la exportacion valida por su cuenta si la base de datos esta lista
            if op != "3":
                preparar_bd()
            acciones[op]()
        else:
            print("Opcion no valida.")


def main():
    menu_principal()


if __name__ == "__main__":
    main()
