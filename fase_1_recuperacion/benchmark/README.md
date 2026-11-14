# Benchmark

El codigo de esta carpeta construye los modelos de recuperacion, ejecuta
consultas y calcula las metricas del benchmark. Las entradas viven en
`data\`; las salidas se escriben en `..\resultados`.
La evaluacion usa la copia local de `evaluate_ir_metrics.py`, basada en el
evaluador entregado dentro de `Material de apoyo\IR evaluation`.

## Módulos del código

- `vector_space_models.py` transforma el corpus en matrices
  documento-término y conserva los modelos vectoriales.
- `information_retrieval.py` realiza la recuperación de documentos a partir
  de una consulta y un modelo persistido.
- `vector_space_models.py` y `benchmark.py` reúnen la preparación de
  representaciones y la ejecución de experimentos del benchmark.
- `evaluate_ir_metrics.py` calcula las métricas a partir de un ranking y sus
  juicios de relevancia.
- `evaluate_benchmark.py` coordina la evaluación de los resultados, mientras
  que `build_report.py` genera informes y resúmenes de las ejecuciones.
- `__init__.py` identifica la carpeta como un paquete de Python y
  `requirements.txt` declara las dependencias específicas de esta etapa.

## Ejecución

El benchmark puede ejecutarse desde la raíz del proyecto con los siguientes
comandos:

```powershell
python "3. Benchmark\codigo\vector_space_models.py"
python "3. Benchmark\main.py" benchmark
```

El comando `benchmark` usa `description` y TF-IDF por defecto. Si no se
indica `--feature`, ejecuta tres configuraciones: unigramas, bigramas y la
combinacion de ambos. Cada ejecucion se guarda por separado en
`resultados\unigram`, `resultados\bigram` y `resultados\both`.

La opción `--representation all` permite generar una comparación de las tres
representaciones utilizadas en la práctica:

```powershell
python "3. Benchmark\main.py" benchmark --representation all
```

Los resultados quedan organizados por feature y representacion, por ejemplo:
`resultados\unigram\frequency` y `resultados\bigram\binary`.

## Archivos de entrada

Los archivos de `data\` definen el corpus de evaluacion:

- `queries.tsv`: TSV con `query_id`, `difficulty_level` y `query`. Contiene
  las 15 consultas del benchmark: cinco faciles, cinco medias y cinco
  dificiles.
- `qrels.tsv`: TSV con `query_id`, `document_id` y `relevance`. Es la tabla
  activa de juicios de relevancia; `relevance=1` indica relevancia parcial y
  `relevance=2` relevancia alta.
- `judgments_review.tsv`: TSV de trabajo para revisar los juicios. Contiene
  la consulta, el documento, el titular, la descripcion, la URL, el nivel de
  relevancia propuesto y `review_status`.
- `judgment_candidates.tsv`: TSV con candidatos adicionales sugeridos por el
  sistema. Incluye `suggested_rank`, `score` y `review_status`; no forma parte
  de `qrels.tsv` hasta que el equipo confirme manualmente esos documentos.
- `criteria.tsv`: TSV que relaciona cada consulta con su `theme` y el
  `criterion` utilizado para definir la relevancia.

El corpus normalizado usado por el motor se encuentra en
`2. Normalizacion\resultados\processed\news_normalized_corpus.tsv`. El motor
relaciona sus filas con los juicios mediante `document_id`.

## Salidas del benchmark

### Salidas directas

Cuando se ejecuta `main.py benchmark`, se generan en `resultados\`:

- `rankings.tsv`: ranking de la ejecucion predeterminada. Contiene
  `query_id`, `document_id`, `rank` y `score`; contiene 150 filas en la
  ejecución predeterminada (15 consultas por 10 documentos).
- `evaluation_results.tsv`: metricas por consulta para ese ranking. Contiene
  `query_id`, `P@10`, `R@10`, `F1@10`, `RR` y `nDCG@10`.
- `comparison.tsv`: promedio global de las configuraciones ejecutadas.
  Contiene `content`, `feature`, `representation`, `P@10`, `R@10`, `F1@10`,
  `RR` y `nDCG@10`.

`content` identifica el campo usado (`title` o `description`); `feature`
identifica unigramas, bigramas o ambos; `representation` identifica
`frequency`, `binary` o `tfidf`.

### Salidas por configuracion

Cada carpeta de `resultados\unigram\`, `resultados\bigram\` y
`resultados\both\` contiene subcarpetas para `frequency`, `binary` y
`tfidf`. Cada subcarpeta conserva:

- `rankings.tsv`: los 10 documentos recuperados para cada consulta.
- `evaluation_results.tsv`: las cinco metricas por consulta.

Estas carpetas permiten comparar experimentos sin sobrescribir los
resultados de otra representacion.

### Modelos persistidos

`resultados\models\title\` y `resultados\models\description\` contienen
archivos `.pkl` generados por `vector_space_models.py`:

- `*_vectorizer.pkl`: el vectorizador de scikit-learn ajustado al corpus.
- `*_matrix.pkl`: la matriz documento-termino dispersa.
- `*_<feature>.pkl`: paquete integrado con vectorizador, matriz,
  vocabulario, dimensiones y estadisticas de dispersion.

El motor de recuperacion carga preferentemente estos paquetes integrados.
Los modelos existen para `frequency`, `binary` y `tfidf`, y para unigramas,
bigramas y la combinacion de ambos.

### Evidencia historica

`resultados\evidence\` conserva una ejecucion documentada anterior. Sus
carpetas contienen, segun la configuracion:

- `rankings.tsv`, `query_results.tsv` y `normalized_queries.tsv`.
- `evaluation_results.tsv` y `evaluator_output.txt`.
- `comparison.tsv` y `manifest.json`.

El `manifest.json` registra configuraciones, versiones, cantidad de
documentos y hashes de entradas de aquella ejecucion. Esta carpeta funciona
como evidencia historica y se mantiene separada de las salidas activas.

`REPORT_STAGE_1.md` tambien es un informe historico; las tablas TSV activas
son la referencia para una ejecucion nueva.

Las salidas de la ejecución predeterminada también se escriben directamente
en `resultados\rankings.tsv` y `resultados\evaluation_results.tsv`.

Tambien puede ejecutarse directamente sobre archivos de ranking:

```powershell
python "3. Benchmark\codigo\evaluate_ir_metrics.py" `
  "3. Benchmark\resultados\rankings.tsv" `
  "3. Benchmark\data\qrels.tsv" `
  --output "3. Benchmark\resultados\evaluation_results.tsv"
```

`judgments_review.tsv` funciona como una hoja de revisión de los juicios de
relevancia: reúne la información de la consulta y del documento, el juicio
propuesto y su estado de revisión para facilitar la comparación y validación
de los documentos. `qrels.tsv` es la tabla compacta de referencia que utiliza
el evaluador; contiene únicamente `query_id`, `document_id` y `relevance`.
Esta separación permite revisar los juicios con contexto sin introducir esas
columnas auxiliares en el cálculo de las métricas.

Se usan cinco temas y tres formulaciones por tema para explorar la variación léxica. Cada tema comparte sus documentos entre niveles, por lo que las 15 consultas no son independientes. La cobertura de juicios es limitada: por convención del evaluador, un documento no juzgado recibe relevancia cero aunque pudiera ser pertinente. Las métricas describen este benchmark y requieren estas limitaciones al interpretarlas.

El manifiesto en `evidence` registra los hashes de las entradas de cada
ejecución para comprobar qué datos produjeron cada resultado. Una ejecución
posterior a un cambio en la tabla de juicios genera resultados comparables y
un informe actualizado. Los juicios representan la relevancia de los
documentos, por lo que su interpretación es independiente de las métricas
obtenidas.
