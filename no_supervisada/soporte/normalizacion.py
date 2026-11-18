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
    texto = texto.translate(COMILLAS)
    texto = re.sub(r"[\U00010000-\U0010ffff]", "", texto)
    texto = re.sub(r"[\r\n\t]+", " ", texto)
    texto = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", texto)
    return " ".join(BOILERPLATE.sub(" ", texto).split())


@lru_cache(maxsize=1)
def cargar_spacy():
    return spacy.load("en_core_web_sm", disable=["parser", "ner"])


def lematizar(doc):
    lemas = []
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
        self.modo = modo

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.modo == "minima":
            return list(X)
        if self.modo == "sin_stopwords":
            return [
                " ".join(token for token in tokenizar(texto) if token not in ENGLISH_STOP_WORDS)
                for texto in X
            ]
        textos = [limpiar_texto(texto) for texto in X]
        if self.modo == "limpieza":
            return textos
        return [lematizar(doc) for doc in cargar_spacy().pipe(textos, batch_size=256)]
