# Agrupamiento No Supervisado de Noticias con K-Means (Tarea B)

Este módulo implementa la **Tarea B (Agrupamiento No Supervisado)** de la Etapa II del proyecto de Procesamiento de Lenguaje Natural, evaluando la capacidad del algoritmo **K-Means** para descubrir asuntos recurrentes, temas emergentes y eventos específicos en el corpus de noticias sin emplear las etiquetas supervisadas durante el ajuste.

---

## 🎯 1. Objetivos y Metodología

* **Algoritmo**: `KMeans` con inicialización `k-means++`, `n_init=10`, `max_iter=300` y semilla fija `random_state=0`.
* **Espacio de Búsqueda Factorial**:
  * Valores de $k \in \{4, 6, 8, 10, 12\}$.
  * Representaciones vectoriales: `TF-IDF` vs `Frecuencia` (Conteos).
  * Normalizaciones: `tokenizado`, `stop_words` y `lematizado`.
  * Reducción de dimensionalidad: Espacio léxico directo vs. proyección semántica latente con `TruncatedSVD` ($d=100$).
* **Métrica Intrínseca de Selección**: **Silhouette Score** sobre la partición `dev`.
* **Evaluación Externa Post-Hoc**:
  * **ARI** (*Adjusted Rand Index*) frente a las categorías temáticas.
  * **NMI** (*Normalized Mutual Information*) frente a las categorías temáticas.
  * Matriz de contingencia (*crosstab*) entre clústeres descubiertos y categorías de origen.
* **Proyección Inversa de Centroides**:
  Para interpretar los tópicos en el espacio latente SVD, se proyectan los centroides hacia el vocabulario original mediante:
  $$\mathbf{w}_k = \boldsymbol{\mu}_k \cdot \mathbf{V}^T$$
  extrayendo las 10 palabras con mayor ponderación semántica.

---

## 📊 2. Entregables Generados

Todos los resultados se encuentran en [`resultados/`](resultados/):
1. [`clustering_results.tsv`](resultados/clustering_results.tsv): Asignación de clústeres para cada noticia en `train`, `dev` y `test` usando `predict()`.
2. [`cluster_interpretation.tsv`](resultados/cluster_interpretation.tsv): Resumen tabular con tamaño de clúster, top 10 palabras características y documentos representativos.
3. [`cluster_interpretation.md`](resultados/cluster_interpretation.md): Reporte cualitativo con titulares de noticias representativas y análisis temático de los 10 eventos descubiertos.
4. [`clustering_experiments.tsv`](resultados/clustering_experiments.tsv): Registro sistemático de todas las configuraciones evaluadas con sus métricas.
5. [`contingency_table.txt`](resultados/contingency_table.txt): Tabla cruzada entre clústeres descubiertos y categorías temáticas originales.

---

## 🚀 3. Instrucciones de Reproducción

Desde la raíz del proyecto (`Proyecto PLN/`):

```powershell
python fase_2_aprendizaje/agrupamiento/train_kmeans.py
```
