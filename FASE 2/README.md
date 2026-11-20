# Fase 2 · Aprendizaje automático aplicado a noticias

Plan de trabajo basado en la [consigna de la Etapa II](Recursos/Proyecto_01_Noticias_Etapa_II.pdf), secciones 3 a 13 y apéndice.

## Alcance

- **Obligatorio:** clasificar noticias por su categoría con al menos tres algoritmos tradicionales.
- **Rama no supervisada elegida:** K-Means para agrupar noticias por asunto o evento. La consigna también ofrece LDA como alternativa.
- Reutilizar los IDs, metadatos, corpus y aplicación de recuperación de la Fase 1. Conservar el benchmark congelado. La Etapa II pide pipelines reproducibles de aprendizaje automático tradicional y excluye LLM y modelos generativos.
- La consigna pide equipos de 3–4 estudiantes. El README de la Fase 1 lista cinco nombres, así que queda por verificar la composición para esta entrega.

## 1. Preparar el corpus y fijar las particiones

El [corpus actual de Fase I](../FASE%201/1.%20Scrapper/news_raw_corpus.tsv) contiene **6,637 noticias**. La preparación de Fase II conserva los 6,637 documentos y los reparte en **4,645 de train, 996 de dev y 996 de test** mediante particiones estratificadas con semilla 0.

- Conservar `id` como `document_id`. Construir `text` concatenando `title` y `description` originales y usar `category` como etiqueta supervisada. Guardar `source`, `date` y `url` como metadatos para inspección.
- Documentar que la categoría proviene del feed RSS y que, según el scraper, se conserva la del primer feed que aporta la noticia. Revisar ejemplos ambiguos y registrar el criterio aplicado. En el análisis de errores, distinguir una confusión del modelo de una noticia que encaja en varias categorías.
- Partir los IDs antes de ajustar el vocabulario, SVD o modelos. Usar `train_test_split` con `shuffle=True`, `random_state=0` y estratificación por categoría cuando sea viable. La proporción sugerida es 70/15/15. Verificar que las particiones sean disjuntas y publicar la distribución por clase en cada una.
- Guardar la asignación en `labeled_documents.tsv` y reutilizarla para la rama no supervisada. Ajustar con `train`, comparar configuraciones en `dev` y reservar `test` para la evaluación final.

## 2. Diseñar experimentos comparables

Variantes de normalización preparadas para ambas tareas:

1. Tokenización mínima como línea base.
2. Tokenización mínima con eliminación de stop words en inglés.
3. Limpieza de los feeds, filtro gramatical y lematización con spaCy.

La corrida `20261111T024437810514Z` contiene **63 configuraciones**: tres normalizaciones, siete combinaciones de unigramas, bigramas y trigramas, y tres representaciones (binaria, frecuencia y TF-IDF). Cada configuración conserva su vocabulario, vectorizador y matrices dispersas por partición. El vocabulario y los pesos IDF se ajustaron con train. Si se usa SVD, ajustarlo también con train y comparar sus dimensiones en dev. [Instrucciones y formato de las salidas](Pipeline%20procesamiento/Programas/README.md).

El entorno de Python se instala con el [requirements.txt del pipeline](Pipeline%20procesamiento/Programas/requirements.txt).

Mantener las mismas particiones, cambiar un factor a la vez en algunas corridas y registrar cada prueba relevante en `experiments.tsv`: ID, tarea, preprocesamiento, representación, n-gramas, reducción, método, hiperparámetros, métricas de dev y decisión de selección. Registrar versiones, semillas y parámetros para reproducir las corridas.

## 3. Clasificación supervisada

- Comparar `MultinomialNB`, `LogisticRegression` y `SVC` como candidatos iniciales. Ajustar hiperparámetros dentro de train y seleccionar en dev por **F1 macro**.
- Guardar para la mejor configuración de cada clasificador en dev las predicciones, accuracy, precision, recall y F1 macro, resultados por clase y matriz de confusión. Revisar al menos diez errores de dev, cuando sea posible, con ID, texto, clase del corpus, predicción y posible causa.
- Congelar el mejor pipeline global, reajustarlo con train + dev y evaluar una sola vez en test. En los TSV de predicciones y métricas, la prueba final lleva `experiment_id=test`.

## 4. Rama no supervisada

Ambas ramas se ajustan con texto de train. Las categorías del corpus sirven únicamente para interpretación externa.

| Rama | Comparaciones | Evidencia |
| --- | --- | --- |
| **K-Means** | Probar varios valores de `k`, inicializaciones, `n_init`, `max_iter` y, si se usa SVD, sus dimensiones. Comparar variantes de Count/TF-IDF, normalización y n-gramas. | Silhouette con espacio y distancia documentados, estabilidad e interpretación en dev cuando sea viable. Tamaño, diez términos del espacio léxico original y al menos dos noticias representativas por grupo. Revisar grupos mixtos, pequeños y sobredimensionados. Añadir ARI, NMI y tabla de contingencia como análisis externo. Asignar dev y test con `predict`. |
| **LDA** | Probar cantidad de temas, `max_iter`, `learning_method` y priors cuando corresponda, además de preprocesamiento, n-gramas y umbrales de vocabulario. | Diez palabras y al menos dos noticias representativas por tema para cada configuración. Mostrar proporciones temáticas de noticias retenidas, perplejidad de train/dev y de test final cuando sea viable. Comparar perplejidad solo con el mismo vocabulario y analizar temas genéricos, solapados o incoherentes. Registrar documentos vacíos y vectores con cero conteos. |

Fijar el criterio de selección en dev antes de evaluar test. Para K-Means, combinar silhouette con la revisión de los grupos y sus noticias. Para LDA, considerar la interpretabilidad y usar perplejidad entre configuraciones con vocabulario comparable.

## 5. Entregables

| Archivo o evidencia | Contenido |
| --- | --- |
| `labeled_documents.tsv` | `document_id`, `text`, `category`, `split` y metadatos útiles. |
| `classification_predictions.tsv` | `document_id`, `gold_label`, `predicted_label`, `split`, `model`, `experiment_id`. |
| `classification_metrics.tsv` | `experiment_id`, `model`, `split`, `accuracy`, `macro_precision`, `macro_recall`, `macro_f1`, con evidencia complementaria por clase y matrices de confusión. |
| `clustering_results.tsv` o `topic_results.tsv` | Según la rama elegida. Clustering incluye `document_id`, `split`, `cluster_id` y configuración. LDA incluye `document_id`, `split`, `topic_id` y `topic_weight`, o proporciones en formato ancho. Adjuntar las tablas de interpretación de la rama. |
| `experiments.tsv` o `experiments.xlsx` | Registro de todas las pruebas relevantes y resumen de configuraciones elegidas. |
| Código, dependencias y comando de inspección | Pipelines, instrucciones reproducibles y una forma de consultar métricas y ejemplos de grupos o temas. |
| Informe PDF | Integrantes, corpus, etiquetas, particiones, controles de fuga, comparación de experimentos, resultados de test, análisis de errores, interpretación no supervisada y limitaciones. |

## Recursos y estructura actual

Los ZIP de clasificación, clustering y temas contienen ejemplos con `20 Newsgroups`. Sirven para adaptar la estructura de los pipelines al corpus propio y al protocolo train/dev/test. El ejemplo de clasificación con SVD termina con un carácter extra que impide ejecutarlo tal como está. Los scripts de polaridad, los léxicos y el lexicón de emojis corresponden a sentimiento, fuera del alcance de las tareas descritas en la consigna.

```text
FASE 2/
  README.md
  Recursos/                 Consigna y materiales de referencia
  Pipeline procesamiento/
    Programas/              Código e instrucciones del pipeline
    resultados/
      1. Combinado/         Variantes combinadas de tokenización, stop words y lematización
      2. Individual/        Variantes individuales con sus representaciones vectoriales
  Kmeans/                   Línea base KM01 y sus resultados
```

El siguiente paso es usar las matrices para comparar los clasificadores y ampliar los experimentos de [K-Means](Kmeans/README.md). La corrida vectorial prepara las entradas de ambas ramas. La selección de configuraciones se hace en dev y la evaluación final en test.
