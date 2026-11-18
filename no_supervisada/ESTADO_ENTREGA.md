# Estado de la entrega de K-Means

La ejecución de esta carpeta está terminada. La segunda revisión aprobó **39 comprobaciones independientes** (resultados/auditoria_cumplimiento.json). Consulta **GUIA_DE_ESTUDIO_NO_SUPERVISADO.pdf** o su versión Markdown para ubicar 23 requisitos y el código exacto que los cubre. Se evaluaron 91 configuraciones y tres ajustes adicionales para estabilidad; se guardó y verificó el modelo final y se generó un informe PDF.

Ganador de selección: **KM083**, sin stopwords, representación binaria, unigramas + bigramas, SVD=50, L2, K-Means con k=12, n_init=10, init=k-means++, max_iter=300 y random_state=0. Silhouette dev de selección: **0.165800**. Tras reajustar el pipeline completo en train + dev, silhouette test final: **0.177708**. No son métricas de clasificación.

| Requisito del PDF | Evidencia en esta carpeta |
| --- | --- |
| Corpus original, IDs y particiones estables | `datos/`, `resultados/labeled_documents.tsv`, validación contra título + descripción |
| Tres normalizaciones y tres representaciones | `resultados/experiments.tsv`, `soporte/normalizacion.py` |
| Unigramas, bigramas y trigramas/combinaciones | Rangos (1,1), (1,2), (1,3) en `experiments.tsv` |
| Variar k, n_init, init y max_iter; semilla reproducible | Bitácora, `plan_experimental.json`, `estabilidad_dev.json` |
| Comparar SVD | Sin SVD, 50, 100 y 150 dimensiones |
| Selección en dev sin etiquetas ni test | `seleccion_dev.json`, código `experimentos.py` |
| Ajustar pipeline final y asignar documentos reservados | `modelos/kmeans_pipeline.joblib`, `clustering_results.tsv` |
| Silhouette, indicando espacio y distancia | `metricas_finales.tsv`: euclidiana en espacio SVD + L2 |
| Tamaños, diez términos y dos noticias por grupo | `cluster_sizes.tsv`, `top_words.tsv`, `representative_documents.tsv` |
| Grupos mixtos, pequeños, grandes y eventos recurrentes | Informe PDF/Markdown, `interpretacion_manual.json`, `eventos_recurrentes.tsv` |
| ARI/NMI y contingencia solo como análisis externo | `metricas_finales.tsv`, `contingencia_*.tsv` |
| Comparaciones controladas e interpretación | `comparaciones_controladas.tsv`, `dev_cluster_interpretation.tsv`, informe |
| Interfaz para inspección | `inspeccionar.py` |
| Código, entorno, semillas y reproducibilidad | Scripts, `requirements-lock.txt`, configuraciones y `ejecucion.log` |
| Informe PDF con figuras, análisis y nombres | `resultados/INFORME_NO_SUPERVISADO.pdf` |
| Integridad de las salidas y carga del modelo | `resultados/validacion.txt`, `validar_resultados.py` |

Se conservaron las 4,645 noticias. El modelo final tiene doce grupos, 120 términos representativos y 24 noticias representativas, además de doce casos de frontera. No se encontraron vectores cero en esta ejecución.

## Limitaciones del resultado

El silhouette es modesto y varios grupos mezclan temas o capturan formato de fuente (Nature/DOI, Continue reading) y referencias temporales. Estas limitaciones están analizadas con documentos reales: la entrega no afirma que los doce grupos sean doce temas puros. Mejorar esa coherencia sería una extensión del estudio, no un resultado ya demostrado.

## Qué queda fuera de esta tarea

Para entregar toda la etapa II, falta que el equipo integre este informe con la parte supervisada y actualice el informe general. Los pendientes de reproducción del módulo supervisado detectados antes (matrices, corrida fija y reajuste del vectorizador final) no se modificaron aquí. Esta entrega completa la rama K-Means; no requiere LDA ni otra ejecución sobre test.

Para compartir, incluye la carpeta completa o `entrega_no_supervisada.zip`: contiene el modelo y las matrices, que el `.gitignore` general excluye de Git.



Autoría: el trabajo fue realizado conjuntamente por todos los integrantes del equipo, según confirmación de la usuaria.
