"""Transformación de texto en inglés para los pipelines de Fase 2."""

import re
from functools import lru_cache

import spacy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS


# Reglas de limpieza y filtrado POS tomadas de la normalización de Fase 1.
BOILERPLATE = re.compile(
    r"\s*continue reading\.{0,3}\s*"
    r"|\s*read more(?: on [^.!?]+)?\s*(?:→|\.{0,3})?\s*"
    r"|\s*sign up(?: here)?[^.!?]*(?:[.!?]|$)"
    r"|\s*subscribe to[^.!?]*(?:[.!?]|$)"
    r"|\s*follow us(?: on [^.!?]+)?[.!?]?\s*"
    r"|\s*click here(?: to [^.!?]+)?[.!?]?\s*"
    r"|\s*read the full (?:article|story)[^.!?]*[.!?]?\s*"
    r"|\s*support the guardian[^.!?]*[.!?]?\s*"
    r"|\s*get our newsletter[^.!?]*[.!?]?\s*"
    r"|\s*delivered to your inbox[^.!?]*[.!?]?\s*",
    flags=re.IGNORECASE,
)
COMILLAS = str.maketrans("’‘´`“”", "''''\"\"")
POS_DESCARTADOS = {"DET", "ADP", "CCONJ", "SCONJ", "PRON", "PART", "AUX", "PUNCT", "SPACE"}
tokenizar = CountVectorizer().build_analyzer()


def limpiar_texto(texto):
    """Elimina ruido, caracteres de control y textos repetitivos."""
    # Se normalizan comillas y caracteres no utilizables.
    texto = texto.translate(COMILLAS)
    texto = re.sub(r"[\U00010000-\U0010ffff]", "", texto)
    texto = re.sub(r"[\r\n\t]+", " ", texto)
    texto = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", texto)
    return " ".join(BOILERPLATE.sub(" ", texto).split())


@lru_cache(maxsize=1)
def cargar_spacy():
    """Carga una sola vez el modelo lingüístico de spaCy."""
    return spacy.load("en_core_web_sm", disable=["parser", "ner"])


def lematizar(doc):
    """Conserva lemas útiles y descarta categorías gramaticales."""
    # Se acumulan los lemas válidos del documento.
    lemas = []
    # Se inspecciona cada token producido por spaCy.
    for token in doc:
        if token.pos_ in POS_DESCARTADOS or token.is_punct or token.is_space:
            continue
        lema = re.sub(r"^[^\w]+|[^\w]+$", "", token.lemma_.lower().strip())
        if lema:
            lemas.append(lema)
    return " ".join(lemas)


class NormalizadorNoticias(TransformerMixin, BaseEstimator):
    """Transformación fija. El aprendizaje del vocabulario ocurre después."""

    def __init__(self, modo="lemas"):
        """Inicializa el normalizador con la variante solicitada."""
        self.modo = modo

    def fit(self, X, y=None):
        """Mantiene la interfaz de scikit-learn sin aprender parámetros."""
        return self

    def transform(self, X):
        """Aplica la limpieza, tokenización o lematización configurada."""
        if self.modo == "minima":
            # La tokenización básica se delega al vectorizador.
            return list(X)
        if self.modo in {"sin_stopwords", "combinado_stopwords"}:
            # Se conservan tokens que no sean stop words.
            textos = [
                " ".join(token for token in tokenizar(texto) if token not in ENGLISH_STOP_WORDS)
                for texto in X
            ]
            return textos
        # Primero se limpia el texto para las variantes lingüísticas.
        textos = [limpiar_texto(texto) for texto in X]
        if self.modo == "limpieza":
            return textos
        if self.modo in {"lemas", "combinado_lemas"}:
            # spaCy analiza y lematiza los textos limpios.
            return [lematizar(doc) for doc in cargar_spacy().pipe(textos, batch_size=256)]
        if self.modo == "tokenizado_lemas":
            # Se tokeniza antes de enviar el texto a spaCy.
            textos = [" ".join(tokenizar(texto)) for texto in textos]
        elif self.modo == "stopwords_lemas":
            # Se quitan stop words antes de lematizar.
            textos = [
                " ".join(token for token in tokenizar(texto) if token not in ENGLISH_STOP_WORDS)
                for texto in textos
            ]
        return [lematizar(doc) for doc in cargar_spacy().pipe(textos, batch_size=256)]
