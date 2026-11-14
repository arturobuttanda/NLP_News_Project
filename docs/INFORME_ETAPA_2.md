# PROYECTO ETAPA II: APRENDIZAJE AUTOMÁTICO APLICADO A NOTICIAS
## Reporte Técnico y Evidencia Experimental — Procesamiento de Lenguaje Natural

**Integrantes del Equipo:**
* Hernandez Espinoza Ricardo
* Espinoza Diaz Diana Karen
* Robles Lozano Byron
* Saenz Buttanda Hector Arturo
* Zurita Salinas Diamanda Abigail

---

## 1. Resumen Ejecutivo del Proyecto

La **Etapa II** del proyecto extiende el sistema de recuperación y procesamiento de noticias desarrollado en la Etapa I, implementando y evaluando de manera sistemática modelos de **Aprendizaje Automático Tradicional** (scikit-learn Pipelines reproducibles, sin modelos generativos ni LLMs). El estudio aborda dos tareas complementarias:

1. **Tarea A (Supervisada - Obligatoria):** Clasificación multiclase de noticias en 10 categorías temáticas mediante tres algoritmos tradicionales: **Multinomial Naive Bayes (`MultinomialNB`)**, **Regresión Logística (`LogisticRegression`)** y **Máquinas de Vectores de Soporte lineales (`LinearSVC`)**, además de variantes con reducción dimensional mediante descomposición en valores singulares truncada (**`TruncatedSVD`**).
2. **Tarea B (No Supervisada - Rama Elegida):** Agrupamiento de documentos mediante **K-Means**, evaluando su capacidad para descubrir asuntos recurrentes, temas emergentes y eventos específicos a partir de la similitud semántica de los textos.

### Protocolo Experimental y Control Estricto de Fuga (Data Leakage)
* **Corpus y Partición:** Se utilizó el corpus ampliado reproducible de la corrida KM01, compuesto por **4,645 noticias periodísticas en inglés** con títulos y descripciones completas.
* **Estratificación Fija:**
  * **Train:** 3,251 noticias (70.0%) — empleado para el ajuste de vocabularios, pesos IDF, transformaciones SVD y parámetros de los modelos.
  * **Dev (Validación):** 697 noticias (15.0%) — reservado estrictamente para la comparación de configuraciones, selección de modelos y afinación de hiperparámetros por **Macro-F1** (supervisado) y **Silhouette Score** (no supervisado).
  * **Test (Evaluación Final):** 697 noticias (15.0%) — congelado e intacto durante toda la fase de desarrollo, evaluado **una sola vez** con el modelo campeón reentrenado sobre la unión de `train + dev`.

---

## 2. Espacio de Representación y Normalización

Se generaron y almacenaron previamente **63 configuraciones vectoriales independientes** para garantizar la comparabilidad controlada:

| Dimensión Experimental | Variantes Evaluadas | Justificación Metodológica |
| :--- | :--- | :--- |
| **Normalización** | 1. `Tokenizado`<br>2. `Stop words`<br>3. `Lematizado` | Comparar desde una línea base mínima hasta un pipeline lingüístico avanzado con filtrado POS y lemas con spaCy (`en_core_web_sm`). |
| **Ponderación Vectorial** | 1. `Binario`<br>2. `Frecuencia`<br>3. `TF-IDF` | Contrastar la mera presencia/ausencia de términos frente a la frecuencia bruta y el peso discriminativo normalizado ($L_2$). |
| **Órdenes de N-gramas** | 1. Unigramas `(1,1)`<br>2. Bigramas `(2,2)`<br>3. Trigramas `(3,3)`<br>4. Combinaciones mixtas | Capturar desde palabras aisladas hasta colocaciones compuestas (*"artificial intelligence"*, *"prime minister"*). |
| **Reducción Dimensional** | `TruncatedSVD` ($d \in \{50, 100, 300\}$) | Compactar la información semántica en espacios densos de menor dimensión (LSA). |

---

## 3. Tarea A: Clasificación Supervisada de Noticias

### 3.1. Comparación de Modelos y Selección en Validación (`dev`)

Se ejecutó un barrido masivo de **398 configuraciones experimentales**, registrando métricas de exactitud, precisión, recall y F1 macro en `classification_metrics.tsv` y `experiments.tsv`:

| ID Exp. | Algoritmo | Normalización | Representación | Hiperparámetros | Accuracy | Macro-Precision | Macro-Recall | **Macro-F1** | Seleccionado |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **C222** | **MultinomialNB** | `stop_words` | `TF-IDF (1,1)` | $\alpha = 0.1$ | 0.7346 | 0.7324 | 0.7158 | **0.7188** | Sí (Mejor NB) |
| **C98** | **LogisticRegression** | `tokenizado` | `TF-IDF (1,1)` | $C = 5.0$, `max_iter=500` | 0.7274 | 0.7274 | 0.7091 | **0.7152** | Sí (Mejor LR) |
| **C239** | **LinearSVC** | `stop_words` | `TF-IDF (1,2)` | $C = 2.0$, `max_iter=2000` | **0.7346** | **0.7296** | **0.7196** | **0.7223** | **Sí (Campeón Global)** |

> **🏆 Decisión de Selección:** El clasificador **`LinearSVC` (Experimento `C239`)** obtuvo el mayor desempeño global en validación con un **Macro-F1 de 0.7223**, superando a Naive Bayes y Regresión Logística gracias a su capacidad para maximizar el margen de separación en espacios dispersos de alta dimensionalidad (128,993 características).

### 3.2. Evaluación Final Independiente en Partición `test`

Siguiendo el protocolo, el pipeline campeón `C239` fue reentrenado con la totalidad de datos disponibles para desarrollo (`train + dev`, $3,251 + 697 = 3,948$ noticias) y evaluado **una única vez** sobre la partición `test`:

| Métrica en Test | Valor Obtenido | Interpretación |
| :--- | :---: | :--- |
| **Accuracy Global** | **73.03%** | 509 de 697 noticias clasificadas correctamente en 10 categorías. |
| **Macro-Precision** | **0.7135** | Alta pureza en las predicciones positivas por categoría. |
| **Macro-Recall** | **0.7147** | Cobertura balanceada a través de todas las clases. |
| **Macro-F1** | **0.7114** | Desempeño armónico y consistente con respecto a `dev` (sin sobreajuste). |

### 3.3. Desempeño Detallado por Clase en `test`

```text
                      precision    recall  f1-score   support

             ciencia     0.8505    0.8198    0.8349       111
cultura_espectaculos     0.7647    0.8553    0.8075        76
            deportes     0.9610    0.9867    0.9737        75
            economia     0.6613    0.5942    0.6260        69
      medio_ambiente     0.7931    0.7797    0.7863        59
    mundo_conflictos     0.5821    0.6393    0.6094        61
            politica     0.5949    0.6528    0.6225        72
               salud     0.6567    0.7719    0.7097        57
            sociedad     0.5250    0.3750    0.4375        56
       tecnologia_ia     0.7455    0.6721    0.7069        61

            accuracy                         0.7303       697
           macro avg     0.7135    0.7147    0.7114       697
        weighted avg     0.7284    0.7303    0.7270       697
```

* **Categorías sobresalientes:** `deportes` (**F1 = 0.9737**), `ciencia` (**F1 = 0.8349**) y `cultura_espectaculos` (**F1 = 0.8075**), debido a que presentan vocabularios altamente especializados y distintivos (*"match, trophy, tournament"*, *"genome, astrophysics, telescope"*).
* **Categorías con mayor desafío:** `sociedad` (**F1 = 0.4375**) y `mundo_conflictos` (**F1 = 0.6094**), caracterizadas por gran heterogeneidad y frontera semántica difusa con política y economía.

---

### 3.4. Análisis Cualitativo de Errores en Validación (`dev`)

Se inspeccionó la totalidad de las 185 predicciones erróneas en `dev` (tasa de error: 26.54%). Las causas fundamentales identificadas son:

1. **Frontera difusa ciencia vs. salud / medio ambiente**:
   * *Ejemplo [N0804]:* Noticia sobre el desarrollo de anticuerpos contra virus respiratorios. Categorizada en el feed como `medio_ambiente`, pero redactada con terminología celular y farmacológica, clasificada por el modelo como `ciencia`.
2. **Intersección geopolítica (política vs. mundo conflictos)**:
   * *Ejemplo [N0120]:* Negociaciones diplomáticas y declaraciones de la Casa Blanca sobre el alto al fuego en Medio Oriente. Asignada a `politica`, el modelo predijo `mundo_conflictos`.
3. **Economía corporativa vs. Tecnología**:
   * *Ejemplo [N0006]:* Reportes de ingresos trimestrales y fluctuaciones bursátiles de fabricantes de chips de inteligencia artificial. Asignada a `politica` / `economia`, predicha como `tecnologia_ia`.
4. **Descripciones telegráficas o incompletas**:
   * Artículos donde el feed RSS solo proveyó 10–15 palabras en la descripción, impidiendo acumular evidencia probabilística suficiente.

---

## 4. Tarea B: Agrupamiento No Supervisado con K-Means

### 4.1. Metodología y Selección de Clústeres

Se ajustaron **50 configuraciones de K-Means** variando $k \in \{4, 6, 8, 10, 12\}$, esquemas de ponderación y reducción latente. Las etiquetas de categoría fueron completamente excluidas del entrenamiento y utilizadas exclusivamente para análisis externo post-hoc.

* **Configuración Seleccionada:** `KMeans(k=10, n_init=10, SVD=100)` sobre `stop_words` + `TF-IDF (1,2)`.
* **Métrica Intrínseca:** **Silhouette Score = 0.2178** (en `dev`).
* **Métricas Externas:** **NMI = 0.1828**, **ARI = 0.0102**.

### 4.2. Interpretación Temática y Descubrimiento de Eventos

Al proyectar los centroides desde el espacio latente SVD al espacio léxico original, K-Means demostró una notable capacidad para aislar **eventos de noticias y asuntos específicos recurrentes**:

| Clúster ID | Tamaño (Train) | Dominante Gold | Top 10 Términos Representativos (Espacio Léxico Original) | Evento / Asunto Descubierto |
| :---: | :---: | :--- | :--- | :--- |
| **#0** | 25 | **deportes (100%)** | *cup, england, world cup, rugby, rugby league, wales, australia* | **Copa Mundial de Rugby** |
| **#2** | 38 | **salud (100%)** | *health, covid, covid 19, countries, global, children, pandemic* | **Políticas de salud pública y pandemia COVID-19** |
| **#5** | 124 | **ciencia (100%)** | *nature, 1038, doi 10, published online, nature published, 2026* | **Publicaciones científicas de la revista Nature** |
| **#6** | 40 | **tecnologia_ia (50%)** | *huang, nvidia, jensen huang, spy, chinese, affair, ceo, chips* | **Caso de espionaje industrial y chips de Nvidia** |
| **#7** | 175 | **politica (36%)** | *trump, ukraine, president, democrats, donald, midterm, russia, war* | **Guerra Ucrania/Rusia y elecciones con Donald Trump** |
| **#8** | 92 | **medio_ambiente (77.2%)** | *climate, cop31, appeared, climate home, news, climate change* | **Cumbre Climática de Naciones Unidas COP31** |
| **#9** | 19 | **tecnologia_ia (100%)** | *codes, promo, promo codes, november, deals, coupon, discounts* | **Ofertas tecnológicas y códigos promocionales** |
| **#3** | 15 | **politica (53.3%)** | *press secretary, white, press, white house, katie zacharia, house* | **Conferencias de prensa de la Casa Blanca** |
| **#4** | 13 | **politica (38.5%)** | *hegseth, impeachment, pete hegseth, articles, massie, secretary* | **Juicio político e investigación a Pete Hegseth** |
| **#1** | 2,710 | **ciencia (14.1%)** | *new, year, ai, years, continue reading, says, scientists, people* | **Clúster residual / Noticias generales de fondo** |

### 4.3. Tabla de Contingencia (Clústeres vs. Categorías Reales)

```text
category    ciencia  cultura_espectaculos  deportes  economia  medio_ambiente  mundo_conflictos  politica  salud  sociedad  tecnologia_ia   All
cluster_id                                                                                                                                     
0                 0                     0        25         0               0                 0         0      0         0              0    25
1               381                   350       320       297             201               229       257    216       234            225  2710
2                 0                     0         0         0               0                 0         0     38         0              0    38
3                 0                     0         0         0               0                 3         8      0         3              1    15
4                 0                     0         0         0               0                 5         5      0         2              1    13
5               124                     0         0         0               0                 0         0      0         0              0   124
6                 0                     0         0        11               0                 5         1      0         3             20    40
7                 5                     1         1        12               7                44        63     10        16             16   175
8                 6                     0         0         3              71                 2         3      0         4              3    92
9                 0                     0         0         0               0                 0         0      0         0             19    19
All             516                   351       346       323             279               288       337    264       262            285  3251
```

* **Pureza de Evento:** Cinco clústeres (#0, #2, #5, #9 y #8) demostraron una pureza temática de entre el **77% y el 100%**, confirmando que el agrupamiento no supervisado descubre agrupaciones de eventos más compactas y específicas que las amplias etiquetas del feed RSS.
* **Estructura Clúster Central vs. Periféricos:** K-Means estructuró el espacio en un "núcleo residual" (Clúster #1) que absorbe noticias misceláneas sin términos repetitivos fuertes, rodeado de clústeres densos y altamente específicos dedicados a noticias de gran repercusión mediática.

---

## 5. Conclusiones y Limitaciones

1. **Eficacia de Modelos Tradicionales:** Los clasificadores lineales sobre matrices dispersas TF-IDF con stop words representan una solución altamente eficiente (entrenamiento en fracciones de segundo) que alcanza **0.7114 de F1 Macro** en un problema de 10 clases desbalanceadas.
2. **Complementariedad Supervisado vs. No Supervisado:** Mientras la clasificación supervisada asigna etiquetas amplias impuestas por la taxonomía de los medios, el agrupamiento con K-Means permite descubrir la verdadera estructura interna de las noticias: **eventos específicos, controversias políticas y coberturas de cumbres internacionales**.
3. **Limitaciones Detectadas:**
   * Las noticias con descripciones de menos de 80 caracteres sufren de escasez léxica (*lexical sparsity*), lo que incrementa el riesgo de confusión con la categoría mayoritaria (`ciencia`).
   * La ausencia de dependencias semánticas no lineales en modelos tradicionales dificulta la resolución de oraciones con negación o ironía, aunque para texto periodístico informativo este fenómeno es marginal.

