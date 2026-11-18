# Pipeline de procesamiento de noticias

El flujo reúne adquisición RSS, normalización y representación vectorial. Prepara las entradas de clasificación supervisada y agrupamiento con K-Means.

## Ejecutar

Desde `Pipeline procesamiento/Programas`, con Python 3.12:

```powershell
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
python pipeline.py
```

Las corridas se ejecutaron con Python 3.12.10 y el modelo de spaCy `en_core_web_sm` 3.8.0.

El pipeline lee `Corpus/news_raw_corpus.tsv` y procesa automáticamente las siete variantes, siete combinaciones de n-gramas y tres representaciones. Para descargar noticias adicionales antes del procesamiento:

```powershell
python pipeline.py --recolectar --dias 14
```

La recolección reutiliza los feeds, filtros y deduplicación del scraper de Fase 1. Conserva los IDs originales y agrega los nuevos con IDs derivados de la URL. Las salidas van a `../resultados/`, separadas en los flujos `1. Combinado` y `2. Individual`. Si se activa la recolección, la copia de `news_raw_corpus.tsv` de esa corrida contiene el corpus ampliado.

Para comparar varias configuraciones sobre una recolección concreta, pasa su copia de `news_raw_corpus.tsv` con `--corpus "ruta/al/news_raw_corpus.tsv"`. Eso reproduce las mismas particiones con la semilla fija.

## Ejecutar todo el procesamiento

Desde esta carpeta, ejecuta el pipeline para generar las siete variantes, sus n-gramas y las representaciones vectoriales:

```powershell
python -B pipeline.py
```

La corrida escribe directamente en las carpetas `N-gramas`, `Binario`, `Frecuencia` y `TD IDF` de cada variante, sin crear carpetas de corrida.

| Carpeta | Procesamiento |
| --- | --- |
| `1. Combinado/1. Ind Tokenizado` | Texto base que se tokeniza al vectorizar. |
| `1. Combinado/1.1 Comb Stop words` | Tokenización y eliminación de stop words. |
| `1. Combinado/1.2 Comb Lematizado` | Limpieza, filtrado POS y lematización. |
| `1. Combinado/1.3 Tokenizado Lematizado` | Limpieza, tokenización y lematización. |
| `1. Combinado/1.4 Stop words Lematizado` | Limpieza, eliminación de stop words y lematización. |
| `2. Individual/2. Ind Stop words` | Solo eliminación de stop words durante la tokenización. |
| `2. Individual/3. Ind Lematizado` | Solo el flujo individual de limpieza, filtrado POS y lematización. |

Cada carpeta contiene un `ngramas.jsonl` con una línea por documento. Cada línea guarda el ID, la partición y tres listas de ocurrencias de unigramas, bigramas y trigramas. Las listas conservan repeticiones y los n-gramas se forman sobre la secuencia de tokens de cada variante. `combinaciones.json` indica qué listas usar para cada una de las siete configuraciones, incluida la combinación de unigramas y trigramas. En la etapa de vectorización, el vocabulario y los pesos se ajustarán a partir de las filas de train.

El pipeline conserva los IDs y las particiones de `N-gramas`. Genera **147 configuraciones** (siete variantes, siete combinaciones y tres representaciones) en `../resultados/<flujo>/<variante>/<representación>/<combinación>/`. Los conteos binarios y de frecuencia usan `CountVectorizer`. TF-IDF usa `TfidfVectorizer` con IDF suavizado y normalización L2. Cada vectorizador se ajusta exclusivamente con las ocurrencias de train.

Las representaciones se guardan directamente dentro de `Binario`, `Frecuencia` y `TD IDF` en las carpetas de cada variante. Contienen 147 configuraciones, cada una con sus matrices de train, dev y test.

Cada combinación contiene `X_train.npz`, `X_dev.npz`, `X_test.npz`, `vocabulario.tsv`, `configuracion.json` y `vectorizador.joblib`. Las filas siguen el orden del `labeled_documents.tsv` de su carpeta de representación, filtrado por partición. Las columnas siguen el orden de `vocabulario.tsv`. Para usar el vectorizador guardado, importa desde `Programas` y pasa una lista de ocurrencias por documento, concatenando las listas de órdenes indicadas en `configuracion.json`. Conserva los n-gramas con espacios como un solo elemento y conserva sus repeticiones.

El texto de entrada es `title + description`. Los documentos se dividen en train/dev/test, aproximadamente 70/15/15, con estratificación y semilla 0. El vocabulario y los pesos IDF se ajustan solo con train. Las etiquetas se usan para estratificar y quedan disponibles como referencia para el análisis posterior.

### Salidas

Cada ejecución escribe bajo `../resultados/<flujo>/<variante>/<representación>/`. Las carpetas de representación son `Binario`, `Frecuencia` y `TD IDF`.

- `news_raw_corpus.tsv` y `labeled_documents.tsv` guardan la entrada, el texto concatenado, las categorías, los IDs y las particiones.
- `corpus_normalizado.tsv` permite revisar el texto que llega al vectorizador.
- `X_train.npz`, `X_dev.npz` y `X_test.npz` son matrices dispersas. Las filas siguen el orden de `labeled_documents.tsv` filtrado por `split`. Las columnas siguen el orden de `vocabulario.tsv`.
- `pipeline_vectorial.joblib` contiene la normalización y el vectorizador ajustado. Cárgalo desde un entorno que tenga esta carpeta disponible para importar `normalizacion`.
- `configuracion.json` registra parámetros, versiones, distribución de categorías y el conteo de vectores vacíos por partición.
