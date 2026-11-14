# Pipeline de Procesamiento y Vectorización de Noticias

Este submódulo reúne la adquisición de noticias RSS, la normalización lingüística multietapa y la construcción de representaciones vectoriales dispersas. Provee la base de datos y matrices para la **clasificación supervisada** y el **agrupamiento con K-Means**.

---

## 🛠️ 1. Componentes del Pipeline

* **`scrapper.py`**: Interfaz de carga del corpus (`news_raw_corpus_4645.tsv`) y módulo de ampliación mediante lectura de feeds RSS sindicados con control de cortesía.
* **`normalizacion.py`**: Clase `NormalizadorNoticias` basada en spaCy (`en_core_web_sm`) con tres modos de preprocesamiento: tokenización mínima, filtrado de *stop words* y lematización morfosintáctica.
* **`ngramas.py`**: Extracción estadística de secuencias de $n$-gramas (unigramas, bigramas y trigramas) y combinación factorial de órdenes.
* **`representaciones.py`**: Generador de matrices dispersas CSR (`scipy.sparse`) bajo esquemas **Binario**, **Frecuencia (TF)** y **TF-IDF** con normalización $L_2$.
* **`pipeline.py`**: Script orquestador principal con interfaz de línea de comandos (CLI).

---

## 🚀 2. Instrucciones de Ejecución

Desde la raíz del proyecto (`Proyecto PLN/`):

```powershell
# Ver todas las opciones disponibles del pipeline CLI
python fase_2_aprendizaje/pipeline/pipeline.py --help

# 1. Preparar las variantes de n-gramas sobre el corpus procesado
python fase_2_aprendizaje/pipeline/pipeline.py --preparar-ngramas

# 2. Vectorizar las combinaciones de n-gramas bajo un identificador de corrida
python fase_2_aprendizaje/pipeline/pipeline.py --vectorizar-ngramas 20261111T024437810514Z
```

---

## 📊 3. Variantes y Espacios Vectoriales Generados

El pipeline genera **63 combinaciones matriciales** organizadas dentro de [`fase_2_aprendizaje/matrices/`](../matrices/):

| Directorio | Tipo de Normalización |
| :--- | :--- |
| **`1 Tokenizado`** | Tokenización básica en minúsculas (palabras de $\ge 2$ caracteres). |
| **`2 Stop words`** | Tokenización con filtrado léxico de palabras vacías en inglés. |
| **`3 Lematizado`** | Limpieza de texto, filtro morfosintáctico (POS) y lematización con spaCy. |

Dentro de cada normalización se calculan las representaciones **Binaria**, **Frecuencia** y **TF-IDF**, cubriendo las 7 combinaciones de órdenes de $n$-gramas (`unigramas`, `bigramas`, `trigramas`, `unigramas_bigramas`, etc.).

Cada carpeta de salida contiene:
* `X_train.npz`: Matriz dispersa de entrenamiento ($3,251$ noticias).
* `X_dev.npz`: Matriz dispersa de desarrollo ($697$ noticias).
* `X_test.npz`: Matriz dispersa de prueba ($697$ noticias).
* `vocabulario.tsv`: Lista ordenada de términos del espacio vectorial.
* `vectorizador.joblib`: Objeto transformador serializado ajustado únicamente en `train`.
* `configuracion.json`: Registro de metadatos, versiones de paquetes y semilla.
