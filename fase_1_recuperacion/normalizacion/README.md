# Normalizacion

Esta carpeta contiene el corpus de entrada y el pipeline que limpia,
lematiza y filtra el texto en ingles con spaCy. El pipeline conserva la
estructura y los metadatos de cada noticia, pero transforma los campos
`title` y `description` para usarlos posteriormente en el benchmark.

`normalize_corpus.py` contiene la implementación del procesamiento y
`requirements.txt` identifica las bibliotecas necesarias para ejecutarlo.
La carpeta `data\raw` representa la fuente original, mientras que
`resultados\processed` representa el corpus preparado para la recuperación de
información.

## Ejecución

El pipeline puede ejecutarse desde la raíz del proyecto con el siguiente
comando:

```powershell
python "2. Normalizacion\codigo\normalize_corpus.py"
```

## Archivos de entrada

`data\raw\news_raw_corpus.tsv` es un archivo TSV con el corpus original. Sus
columnas son:

- `id`: identificador unico de la noticia.
- `title`: titular original.
- `description`: descripcion o cuerpo resumido original.
- `source`: medio de comunicacion.
- `category`: categoria tematica.
- `date`: fecha de publicacion.
- `url`: enlace de la noticia.
- `retrieved_at`: fecha en que fue recuperada.

## Archivos de salida

El script genera `resultados\processed\news_normalized_corpus.tsv`, también en
formato TSV y con las mismas columnas que el corpus original. Este archivo es
el corpus limpio que utiliza el Benchmark para construir los modelos de
recuperación y conservar la correspondencia entre cada documento y su
identificador original. Los campos
`id`, `source`, `category`, `date`, `url` y `retrieved_at` conservan la
identidad y los metadatos de la noticia. Los campos `title` y `description`
contienen el texto procesado:

1. Se eliminan fragmentos de navegacion o suscripcion de algunos feeds.
2. Se tokeniza el texto con spaCy.
3. Se eliminan categorias gramaticales funcionales y signos de puntuacion.
4. Se convierten las palabras a minusculas y se lematizan.

Esta es la salida activa que utiliza el benchmark. El archivo mantiene una
fila por documento y permite relacionar cada texto normalizado con su `id`
original.

`resultados\processed\news_normalized_corpus_en.tsv` es un archivo alternativo
con la misma estructura y contenido normalizado en inglés. Se conserva por
compatibilidad con ejecuciones anteriores, aunque el script actual no lo
genera ni lo actualiza y el Benchmark no lo utiliza. La salida vigente para
reproducir la evaluación es `news_normalized_corpus.tsv`.

El pipeline elimina fragmentos de navegacion y suscripcion propios de algunos
feeds, como `Continue reading`, `Read more`, `Sign up`, `Subscribe` y
`Follow us`, antes de aplicar tokenizacion, lematizacion y filtrado POS.
