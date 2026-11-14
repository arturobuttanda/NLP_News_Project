# Sistema de Recuperación de Información y Aprendizaje Automático sobre Noticias (PLN)

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange.svg)](https://scikit-learn.org/)
[![spaCy](https://img.shields.io/badge/spaCy-3.7%2B-09a3d5.svg)](https://spacy.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Este repositorio contiene la implementación integral, reproducible y documentada del proyecto de **Procesamiento de Lenguaje Natural (PLN)**, estructurado en dos grandes etapas:

1. **Fase 1 · Recuperación de Información (IR) y Modelos de Espacio Vectorial (VSM)**: Adquisición de noticias por feeds RSS, normalización lingüística multietapa, indexación vectorial (31 configuraciones) y evaluación con juicios de relevancia humanos sobre 16 consultas temáticas ($P@10$, $R@10$, $\text{MAP}$, $\text{nDCG}@10$).
2. **Fase 2 · Aprendizaje Automático Tradicional sobre Noticias**:
   * **Pipeline Vectorial (63 espacios)**: Matrices dispersas combinando 3 normalizaciones (Tokenizado, Stop words, Lematizado) $\times$ 3 ponderaciones (Binario, Frecuencia, TF-IDF) $\times$ $n$-gramas (unigramas a trigramas).
   * **Tarea A (Supervisada)**: Clasificación temática multiclase (10 categorías) evaluando 398 configuraciones mediante `Pipeline`s de Scikit-Learn y `GridSearchCV`. Modelo campeón: `LinearSVC(C=2.0)` con **Macro-F1 = 0.7114** y **Accuracy = 73.03%** en el conjunto de prueba ciego (`test`).
   * **Tarea B (No Supervisada)**: Agrupamiento con `K-Means` ($k=10$) sobre espacio latente `TruncatedSVD(100)`, proyección de centroides ($\mathbf{w}_k = \boldsymbol{\mu}_k \mathbf{V}^T$) y descubrimiento de 10 eventos noticiosos reales sin uso de etiquetas.

---

## 👥 Integrantes del Equipo

* **Hernandez Espinoza Ricardo**
* **Espinoza Diaz Diana Karen**
* **Robles Lozano Byron**
* **Saenz Buttanda Hector Arturo**
* **Zurita Salinas Diamanda Abigail**

---

## 📁 Arquitectura del Repositorio

```text
Proyecto PLN/
├── README.md                           # Documentación principal del repositorio (este archivo)
├── requirements.txt                    # Dependencias unificadas de Python (Fase 1 y 2)
├── .gitignore                          # Exclusión estricta de binarios pesados (NPZ, JOBLIB, DB)
├── demostracion_en_vivo.ps1            # CLI interactivo de PowerShell para defensas y demostraciones
├── data/                               # Datos centralizados del proyecto
│   ├── raw/                            # noticias.db, news_raw_corpus_2157.tsv, news_raw_corpus_4645.tsv
│   └── processed/                      # labeled_documents.tsv (4,645 noticias particionadas 70/15/15)
├── fase_1_recuperacion/                # Fase 1: Recuperación de Información (IR)
│   ├── README.md                       # Documentación específica de Fase 1
│   ├── scraper/                        # Recolector RSS, politesse y almacenamiento SQLite
│   ├── normalizacion/                  # Pipeline morfosintáctico con spaCy (POS filtering y lematización)
│   └── benchmark/                      # Motor VSM (31 modelos), CLI interactivo y juicios de relevancia
├── fase_2_aprendizaje/                 # Fase 2: Aprendizaje Automático
│   ├── README.md                       # Resumen ejecutivo de la Fase 2
│   ├── pipeline/                       # Pipeline de procesamiento, extracción de n-gramas y vectorización
│   │   ├── README.md                   # Uso de pipeline.py y catálogo de parámetros
│   │   ├── pipeline.py                 # Orquestador maestro CLI
│   │   ├── normalizacion.py            # Normalizador lingüístico
│   │   ├── ngramas.py                  # Extractor de n-gramas
│   │   ├── representaciones.py         # Generador de matrices dispersas CSR
│   │   └── scrapper.py                 # Conector y deduplicador de corpus
│   ├── matrices/                       # 63 espacios vectoriales precalculados por partición
│   │   ├── 1 Tokenizado/               # Tokenización básica (Binario, Frecuencia, TF-IDF)
│   │   ├── 2 Stop words/               # Tokenización sin palabras vacías
│   │   └── 3 Lematizado/               # Lematización profunda con spaCy
│   ├── clasificacion/                  # Tarea A: Clasificación Supervisada
│   │   ├── README.md                   # Documentación exhaustiva función por función
│   │   ├── train_classifiers.py        # Auditoría de etiquetas, GridSearchCV y barrido de 398 modelos
│   │   ├── evaluate_final.py           # Reentrenamiento en train+dev y evaluación ciega en test
│   │   ├── error_analysis.py           # Diagnóstico cualitativo de 12 casos de error en dev
│   │   └── resultados/                 # experiments.tsv, metrics.tsv, predictions.tsv, confusion_matrices
│   └── agrupamiento/                   # Tarea B: Clustering No Supervisado
│       ├── README.md                   # Documentación de K-Means y proyección latente
│       ├── train_kmeans.py             # Entrenamiento de K-Means, proyección SVD y contingencia
│       └── resultados/                 # clustering_results.tsv, cluster_interpretation.tsv/md
├── docs/                               # Documentación formal y entregables
│   ├── presentacion_proyecto_pln.html  # Dashboard interactivo con Chart.js para defensas
│   ├── INFORME_ETAPA_2.md              # Informe técnico detallado de la Etapa II
│   └── PLAN_ESTRUCTURACION.md          # Bitácora de ingeniería y diseño de software
└── recursos/                           # Guías oficiales docentes y scripts didácticos de referencia
    ├── guias_docente/                  # PDFs de especificación (Etapa I y Etapa II)
    └── scripts_didacticos/             # Scripts de referencia (APIs, clasificación, clustering)
```

---

## ⚡ Instalación y Puesta en Marcha

### 1. Clonar el Repositorio
```bash
git clone https://github.com/usuario/proyecto-pln.git
cd proyecto-pln
```

### 2. Crear y Activar un Entorno Virtual
```powershell
# En Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# En Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar Dependencias Unificadas
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Descargar el Modelo Lingüístico de spaCy
```bash
python -m spacy download en_core_web_sm
```

---

## 🔍 Fase 1: Recuperación de Información (IR)

La Fase 1 implementa un motor completo de recuperación de noticias basado en **Modelos de Espacio Vectorial (VSM)**:

* **Corpus Base:** 2,157 artículos periodísticos en inglés clasificados en 10 categorías temáticas, recolectados mediante feeds RSS de medios internacionales (BBC, The Guardian, NPR, Nature, TechCrunch, ESPN, etc.).
* **Preprocesamiento Lingüístico:** Conversión a minúsculas, filtrado de URLs y etiquetas HTML, eliminación de signos de puntuación, filtrado selectivo por categorías gramaticales (POS Tagging: sustantivos, verbos, adjetivos y adverbios) y lematización con spaCy.
* **Espacio Vectorial:** Construcción de 31 combinaciones de esquemas de ponderación (Booleano, Frecuencia de Término pura, Sublineal $1 + \log(TF)$, TF-IDF estándar y variantes normalizadas $L_2$).
* **Evaluación Formal:** Benchmark sobre 16 consultas de prueba evaluadas contra juicios de relevancia humanos, reportando:
  * Precisión en los primeros 10 resultados ($P@10$)
  * Recall en los primeros 10 resultados ($R@10$)
  * F1-Score en los primeros 10 resultados ($F1@10$)
  * Reciprocal Rank ($RR$)
  * Normalized Discounted Cumulative Gain ($nDCG@10$)

### Comandos de Fase 1
```powershell
# Búsqueda interactiva en el corpus:
python fase_1_recuperacion/benchmark/main.py search --query "artificial intelligence healthcare regulation"

# Ejecutar el benchmark completo de evaluación:
python fase_1_recuperacion/benchmark/main.py benchmark
```

---

## 🤖 Fase 2: Aprendizaje Automático sobre Noticias

La Fase 2 reutiliza el corpus ampliado y congelado de **4,645 noticias**, implementando protocolos científicos reproducibles para evitar cualquier fuga de datos (*data leakage*).

### Protocolo de Partición Estratificada (70% / 15% / 15%)
El conjunto se dividió con semilla fija (`random_state=0`) y estratificación estricta por categoría:
* **`train` (3,251 noticias - 70%):** Utilizado exclusivamente para aprender vocabularios, pesos IDF, reducciones SVD y ajustar los clasificadores.
* **`dev` (697 noticias - 15%):** Utilizado para validación cruzada, sintonización de hiperparámetros, comparación de modelos y análisis cualitativo de errores.
* **`test` (697 noticias - 15%):** Conjunto ciego reservado para una única evaluación final.

---

### Tarea A: Clasificación Supervisada Multiclase

Se compararon 4 familias de algoritmos tradicionales a través de dos metodologías complementarias:

1. **Sintonización formal con `Pipeline` + `GridSearchCV`:**
   Implementado en [`train_classifiers.py`](fase_2_aprendizaje/clasificacion/train_classifiers.py), ejecuta validación cruzada de 5 pliegues (`StratifiedKFold(5)`) optimizando `scoring="f1_macro"` directamente sobre los textos crudos de `train`.
2. **Barrido exhaustivo factorial de 398 configuraciones:**
   Explora las matrices precalculadas variando normalización (`tokenizado`, `stop_words`, `lematizado`), ponderación (`binary`, `frecuencia`, `tfidf`), órdenes de $n$-gramas e hiperparámetros ($\alpha$ para Naive Bayes, $C$ para LinearSVC y LogisticRegression, $d$ para TruncatedSVD).

#### Tabla Comparativa Oficial de Modelos Seleccionados (`dev` y `test`)

| Experimento | Algoritmo | Normalización | Representación | Hiperparámetros | Partición | Accuracy | Macro-Precision | Macro-Recall | **Macro-F1** |
| :---: | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **C222** | MultinomialNB | `stop_words` | `tfidf unigramas` | $\alpha = 0.1$ | `dev` | 0.7346 | 0.7324 | 0.7158 | **0.7188** |
| **C98** | LogisticRegression | `tokenizado` | `tfidf unigramas` | $C = 5.0$ | `dev` | 0.7274 | 0.7274 | 0.7091 | **0.7152** |
| **C239** | **LinearSVC** *(Campeón Dev)* | `stop_words` | `tfidf unig+big` | $C = 2.0$ | `dev` | **0.7346** | **0.7296** | **0.7196** | **0.7223** |
| — | RandomForest + SVD | `stop_words` | `tfidf + SVD(100)` | $n=100, d=15$ | `dev` | 0.6499 | 0.6380 | 0.6270 | **0.6325** |
| **test** | **LinearSVC** *(Evaluación Final)* | `stop_words` | `tfidf unig+big` | $C = 2.0$ | `test` | **0.7303** | **0.7135** | **0.7147** | **0.7114** |

> **🏆 Modelo Ganador:** `LinearSVC(C=2.0)` reentrenado sobre `train + dev` (3,948 noticias) y evaluado **una sola vez** sobre `test` (697 noticias), alcanzando **73.03% de exactitud** y **0.7114 de Macro-F1**.

#### Diagnóstico Cualitativo de Errores
Se analizaron en detalle los 185 errores en `dev` (26.54% de tasa de error) en [`error_analysis.md`](fase_2_aprendizaje/clasificacion/resultados/error_analysis.md). Las causas principales radican en:
* **Frontera difusa ciencia / medio ambiente:** Artículos sobre cambio climático y biodiversidad con terminología biológica.
* **Intersección geopolítica:** Declaraciones de mandatarios en contextos bélicos compartiendo léxico entre `politica` y `mundo_conflictos`.
* **Noticias corporativas:** Reportes sobre ganancias de empresas tecnológicas clasificados en `economia` en vez de `tecnologia_ia`.

---

### Tarea B: Agrupamiento No Supervisado con K-Means

* **Configuración Ganadora:** `KMeans(k=10)` sobre `TF-IDF + TruncatedSVD(100)` con normalización `stop_words` (Silhouette = **0.8394**).
* **Fórmula de Proyección Inversa de Centroides:** Para interpretar los tópicos en el espacio latente SVD, los centroides latentes $\boldsymbol{\mu}_k \in \mathbb{R}^{100}$ se proyectan hacia el vocabulario léxico original mediante:
  $$\mathbf{w}_k = \boldsymbol{\mu}_k \cdot \mathbf{V}^T$$
  extrayendo los 10 términos de mayor peso semántico.
* **10 Eventos Reales Descubiertos Sin Supervisión:**
  1. *Caso judicial Genaro García Luna en Nueva York*
  2. *Fútbol Europeo, Premier League y Champions League*
  3. *Accidente y rehabilitación de la Línea 12 del Metro CDMX*
  4. *Guerra en Ucrania y corredor de granos del Mar Negro*
  5. *Misión lunar Artemis de la NASA y exploración espacial*
  6. *Salud pública, variantes post-COVID y vacunación*
  7. *Inflación, tasas de interés y decisiones de Banxico*
  8. *Crisis climática, estrés hídrico y cumbres COP*
  9. *Reformas constitucionales y política gubernamental*
  10. *Premios cinematográficos, estrenos y cultura global*

---

## 🖥️ Comandos de Ejecución y Reproducción

### Ejecutar Todo el Flujo Supervisado en 1 Línea
```powershell
python fase_2_aprendizaje/clasificacion/train_classifiers.py; python fase_2_aprendizaje/clasificacion/error_analysis.py; python fase_2_aprendizaje/clasificacion/evaluate_final.py
```

### Ejecutar el Agrupamiento No Supervisado
```powershell
python fase_2_aprendizaje/agrupamiento/train_kmeans.py
```

### Abrir la Demostración Interactiva en Vivo
```powershell
.\demostracion_en_vivo.ps1
```
*(Permite abrir el Dashboard HTML, ejecutar búsquedas interactivas en vivo o consultar métricas al instante).*

---

## 📑 Dashboard Interactivo de Resultados

Para visualizar los gráficos de barras interactivos (Chart.js), diagramas de flujo metodológicos, matrices de confusión y fichas técnicas archivo por archivo, abre en tu navegador:

👉 [`docs/presentacion_proyecto_pln.html`](docs/presentacion_proyecto_pln.html)

---

## 📜 Licencia y Uso Académico

Este proyecto ha sido desarrollado con fines exclusivamente académicos para el curso de **Procesamiento de Lenguaje Natural (PLN)**. Código publicado bajo la licencia [MIT](LICENSE).
