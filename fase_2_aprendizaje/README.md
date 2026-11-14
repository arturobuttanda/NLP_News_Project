# Fase 2 · Aprendizaje Automático Aplicado a Noticias

Este directorio implementa de forma integral y reproducible la **Etapa II** del proyecto de Procesamiento de Lenguaje Natural, siguiendo las especificaciones oficiales de la cátedra ([`recursos/guias_docente/Proyecto_01_Noticias_Etapa_II.pdf`](../recursos/guias_docente/Proyecto_01_Noticias_Etapa_II.pdf)).

---

## 🎯 1. Alcance y Objetivos de la Etapa II

1. **Tarea A (Obligatoria) — Clasificación Supervisada Multiclase**:
   * Predecir la categoría temática de las noticias (`politica`, `economia`, `tecnologia_ia`, `ciencia`, `salud`, `medio_ambiente`, `deportes`, `mundo_conflictos`, `cultura_espectaculos`, `sociedad`).
   * Comparación sistemática de algoritmos tradicionales (`MultinomialNB`, `LogisticRegression`, `LinearSVC` y la variante `RandomForestClassifier + SVD`).
   * Sintonización con Scikit-Learn `Pipeline`s y `GridSearchCV` sobre la partición `train`.
   * Barrido controlado de **398 configuraciones** experimentales sobre las matrices del proyecto.
   * Criterio primario de selección: **Macro-F1** en validación (`dev`).
   * Reentrenamiento estricto del campeón en `train + dev` (3,948 noticias) y evaluación única sobre `test` (697 noticias).

2. **Tarea B (Elegida) — Agrupamiento No Supervisado con K-Means**:
   * Descubrimiento de temas emergentes, asuntos recurrentes y eventos de noticias sin utilizar las etiquetas de clase durante el ajuste.
   * Exploración de hiperparámetros: $k \in \{4, 6, 8, 10, 12\}$, esquemas de vectorización (TF-IDF vs Conteos), y reducción de dimensionalidad latente con `TruncatedSVD` ($d=100$).
   * Selección por coeficiente de silueta (*Silhouette Score*) en `dev`.
   * Proyección inversa de centroides al vocabulario original ($\mathbf{w}_k = \boldsymbol{\mu}_k \mathbf{V}^T$), interpretación cualitativa de clústeres y matriz de contingencia externa.

---

## 📁 2. Estructura del Módulo

```text
fase_2_aprendizaje/
├── README.md               # Esta guía general de la Fase 2
├── pipeline/               # Ingesta, normalización lingüística y exportador vectorial
│   ├── pipeline.py         # Orquestador principal CLI de vectorización
│   ├── normalizacion.py    # Preprocesamiento, lematización con spaCy y filtros
│   ├── ngramas.py          # Extractor de unigramas, bigramas y trigramas
│   ├── representaciones.py # Generador de matrices dispersas (Binario, TF, TF-IDF)
│   └── scrapper.py         # Integración y deduplicación con el scraper de Fase 1
├── matrices/               # 63 espacios vectoriales precalculados
│   ├── 1 Tokenizado/       # Tokenización básica (Binario, Frecuencia, TF-IDF)
│   ├── 2 Stop words/       # Tokenización sin palabras vacías
│   └── 3 Lematizado/       # Normalización lematizada completa
├── clasificacion/          # Tarea A: Clasificación Supervisada
│   ├── README.md           # Documentación técnica exhaustiva función por función
│   ├── train_classifiers.py# Auditoría, GridSearchCV y barrido de 398 modelos
│   ├── evaluate_final.py   # Reajuste en train+dev y evaluación ciega en test
│   ├── error_analysis.py   # Diagnóstico cualitativo de 12 casos de error en dev
│   └── resultados/         # TSVs oficiales de métricas, predicciones y bitácora
└── agrupamiento/           # Tarea B: Clustering No Supervisado
    ├── README.md           # Documentación del modelo de agrupamiento K-Means
    ├── train_kmeans.py     # Entrenamiento, proyección latente y contingencia
    └── resultados/         # Clústeres, interpretación y tabla de contingencia
```

---

## 🔬 3. Corpus y Particiones Disjuntas (Control Anti-Fuga)

El corpus base procesado contiene **4,645 noticias** con 10 categorías canónicas, dividido de forma estratificada con `train_test_split(shuffle=True, random_state=0, stratify=category)` en [`data/processed/labeled_documents.tsv`](../data/processed/labeled_documents.tsv):

| Partición | Proporción | Cantidad de Noticias | Uso Metodológico Estricto |
| :--- | :---: | :---: | :--- |
| **`train`** | 70% | **3,251** | Ajuste de vocabularios, IDF, SVD y modelos base. |
| **`dev`** (desarrollo) | 15% | **697** | Selección de hiperparámetros, comparación y análisis de errores. |
| **`test`** (prueba) | 15% | **697** | Evaluación final ciega única. *Nunca se usa para ajustar ni seleccionar.* |

> **Garantía Anti-Fuga (*Data Leakage Control*):** Todos los transformadores (`CountVectorizer`, `TfidfVectorizer`, `TruncatedSVD`) se ajustan exclusivamente con `fit_transform` sobre `train`, y únicamente aplican `transform` sobre `dev` y `test`.

---

## 📊 4. Resumen de Resultados Oficiales

### A. Tarea A: Clasificación Supervisada (Ganador Global)
* **Modelo Campeón:** `LinearSVC(C=2.0)` sobre normalización `stop_words` con `TF-IDF unigramas_bigramas` (Experimento `C239`).
* **Desempeño en Validación (`dev`):** **Macro-F1 = 0.7223** | **Accuracy = 73.46%**
* **Evaluación Final en Prueba (`test`):**
  * **Exactitud (Accuracy):** **`73.03%`**
  * **Macro-Precision:** **`0.7135`**
  * **Macro-Recall:** **`0.7147`**
  * **Macro-F1:** **`0.7114`**

### B. Tarea B: Clustering No Supervisado (K-Means)
* **Configuración Seleccionada:** `KMeans(k=10)` sobre `TF-IDF + TruncatedSVD(100)` con normalización `stop_words` (Experimento `U29`).
* **Métrica Intrínseca:** **Silhouette Score = 0.8394**
* **Eventos Descubiertos Sin Etiquetas:**
  1. *Juicio a Genaro García Luna en Nueva York*
  2. *Fútbol Europeo y Champions League*
  3. *Accidente y Reapertura de Línea 12 del Metro CDMX*
  4. *Guerra en Ucrania y Bloqueo de Granos del Mar Negro*
  5. *Carrera Espacial y Misión Lunar Artemis de la NASA*
  6. *Salud Pública y Vacunación Post-COVID*
  7. *Inflación, Tasas de Interés y Política Monetaria de Banxico*
  8. *Crisis Climática, Sequías y COP sobre Biodiversidad*
  9. *Conflictos y Reformas del Gobierno en Política Nacional*
  10. *Cine, Premios de la Academia y Entretenimiento Internacional*

---

## 🚀 5. Instrucciones de Reproducción

Desde la raíz del repositorio (`Proyecto PLN/`):

```powershell
# 1. Ejecutar el pipeline de clasificación supervisada (Tarea A)
python fase_2_aprendizaje/clasificacion/train_classifiers.py
python fase_2_aprendizaje/clasificacion/error_analysis.py
python fase_2_aprendizaje/clasificacion/evaluate_final.py

# 2. Ejecutar el pipeline de clustering no supervisado (Tarea B)
python fase_2_aprendizaje/agrupamiento/train_kmeans.py
```
