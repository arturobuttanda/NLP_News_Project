# PROJECT STAGE 1: NEWS RETRIEVAL REPORT

> **Nota de vigencia:** este documento es un informe histórico conservado como
> evidencia. Las métricas y rankings vigentes se encuentran en
> `rankings.tsv`, `evaluation_results.tsv` y `comparison.tsv`, regenerados por
> el código actual. No uses las cifras de este informe para comparar una nueva
> ejecución.

## Natural Language Processing — Final Evidence & Evaluation Report

**Integrantes del Equipo:**
* [Nombre de Integrante 1]
* [Nombre de Integrante 2]
* [Nombre de Integrante 3]

---

## 1. Resumen Ejecutivo del Proyecto

Este reporte documenta la implementación completa y evaluación cuantitativa del sistema de recuperación de noticias (*Information Retrieval System*) basado en el **Modelo de Espacio Vectorial (Vector Space Model - VSM)** y **Similitud de Cosenos**, cubriendo las especificaciones de las Secciones 4 a 13 del documento oficial.

### Pipeline del Sistema:
1. **Adquisición del Corpus**: 2,157 artículos periodísticos internacionales en inglés provenientes de 45 medios independientes (*The Guardian, BBC News, The Hindu, CNBC, ScienceDaily, Nature, ESPN, NPR, Wired, etc.*), cubriendo 10 categorías temáticas sin valores nulos.
2. **Normalización de Texto**: Pipeline lingüístico mediante spaCy (`en_core_web_sm`) aplicando tokenización, lematización en minúsculas, filtrado riguroso de stopwords funcionales (`DET`, `ADP`, `CCONJ`, `SCONJ`, `PRON`, `PART`, `AUX`, `PUNCT`, `SPACE`) y sanitización de caracteres de control y emojis.
3. **Representaciones Vectoriales**: Generación de matrices dispersas (`scipy.sparse.csr_matrix`) para `Title` y `Description` en 3 representaciones (*Frequency*, *Binary*, *TF-IDF*) con unigramas y bigramas (`ngram_range=(1, 2)`).
4. **Motor de Recuperación**: Transformación en tiempo real de consultas en texto libre bajo el mismo pipeline de normalización, proyección vectorial y cálculo de similitud de cosenos para rankear los Top-10 documentos más relevantes.
5. **Benchmark Cuantitativo**: Creación de un conjunto de 15 consultas (5 Easy, 5 Medium, 5 Hard) con juicios de relevancia anotados (*qrels* con 180 anotaciones en escala 1 y 2) y evaluación formal de 5 métricas: **P@10, R@10, F1@10, MRR y nDCG@10**.

---

## 2. Modelos de Espacio Vectorial Generados

| Campo | Representación | Dimensión de Matriz ($N \times D$) | Unigramas | Bigramas | Features Totales | Dispersión (Sparsity) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`title`** | **Frequency** (Count) | $2,157 \times 21,247$ | 6,271 | 14,976 | **21,247** | 99.92% |
| **`title`** | **Binary** (One-Hot) | $2,157 \times 21,247$ | 6,271 | 14,976 | **21,247** | 99.92% |
| **`title`** | **TF-IDF** | $2,157 \times 21,247$ | 6,271 | 14,976 | **21,247** | 99.92% |
| **`description`** | **Frequency** (Count) | $2,157 \times 66,052$ | 12,670 | 53,382 | **66,052** | 99.92% |
| **`description`** | **Binary** (One-Hot) | $2,157 \times 66,052$ | 12,670 | 53,382 | **66,052** | 99.92% |
| **`description`** | **TF-IDF** | $2,157 \times 66,052$ | 12,670 | 53,382 | **66,052** | 99.92% |

---

## 3. Resultados Cuantitativos del Benchmark (15 Consultas)

Evaluación generada con el modelo **TF-IDF sobre `Description`** (Unigramas + Bigramas), evaluando las predicciones almacenadas en `3. Benchmark/resultados/rankings.tsv` frente a `3. Benchmark/data/qrels.tsv`:

| Query ID | Nivel | Texto de la Consulta | P@10 | R@10 | F1@10 | RR (MRR) | nDCG@10 |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **Q01** | 1 (Easy) | new government regulations for artificial intelligence | 0.1000 | 0.0833 | 0.0909 | 0.1111 | 0.0810 |
| **Q02** | 1 (Easy) | extreme weather events caused by climate change | **0.2000** | **0.1667** | **0.1818** | **1.0000** | **0.3037** |
| **Q03** | 1 (Easy) | presidential election campaign and voter polling results | 0.1000 | 0.0833 | 0.0909 | 0.1429 | 0.0958 |
| **Q04** | 1 (Easy) | space exploration missions and astronomical discoveries | 0.1000 | 0.0833 | 0.0909 | 0.2500 | 0.0386 |
| **Q05** | 1 (Easy) | football match scores and championship tournament results | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **Q06** | 2 (Med) | surging cost of food and grocery inflation | 0.1000 | 0.0833 | 0.0909 | 0.2500 | 0.1158 |
| **Q07** | 2 (Med) | digital cybersecurity attacks and data network breaches | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **Q08** | 2 (Med) | renewable clean energy transition and solar power projects | **0.2000** | **0.1667** | **0.1818** | **0.3333** | **0.0795** |
| **Q09** | 2 (Med) | clinical trials for novel cancer treatment therapies | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **Q10** | 2 (Med) | electric vehicle market adoption and battery manufacturing | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **Q11** | 3 (Hard) | companies replacing workers with artificial intelligence | 0.1000 | 0.0833 | 0.0909 | 0.1429 | 0.0299 |
| **Q12** | 3 (Hard) | geopolitical conflicts disrupting global energy supplies | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **Q13** | 3 (Hard) | coastal erosion and rising sea levels threatening settlements | **0.1000** | **0.0833** | **0.0909** | **1.0000** | **0.2873** |
| **Q14** | 3 (Hard) | epidemiological containment and viral outbreak surveillance | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **Q15** | 3 (Hard) | monopolistic antitrust actions against big tech corporations | 0.1000 | 0.0833 | 0.0909 | 0.1250 | 0.0849 |
| **PROMEDIO**| — | **Métricas Globales del Benchmark (15 Consultas)** | **0.0733** | **0.0611** | **0.0667** | **0.2237** | **0.0744** |

### Resumen de Métricas Agregadas:
* **Mean Precision@10 (P@10)**: `0.0733`
* **Mean Recall@10 (R@10)**: `0.0611`
* **Mean F1@10**: `0.0667`
* **Mean Reciprocal Rank (MRR)**: `0.2237`
* **Mean nDCG@10**: `0.0744`

### Desglose por Nivel de Complejidad:
* **Nivel 1 (Easy - Fuerte solapamiento léxico)**: P@10 = `0.1000`, MRR = `0.3008`, nDCG@10 = `0.1038`.
* **Nivel 2 (Medium - Variación léxica / sinonimia)**: P@10 = `0.0600`, MRR = `0.1167`, nDCG@10 = `0.0391`.
* **Nivel 3 (Hard - Relación semántica profunda sin solapamiento)**: P@10 = `0.0600`, MRR = `0.2536`, nDCG@10 = `0.0804`.

---

## 4. Análisis de Errores (Sección 12 del PDF)

Tal como exige la Sección 12, se analizan a profundidad **5 consultas con alto rendimiento** y **5 consultas con bajo rendimiento**, explicando las causas lingüísticas y las limitaciones intrínsecas del modelo VSM léxico frente a futuros modelos neuronales basados en embeddings.

### A. Consultas con Alto Rendimiento (Top 5)

1. **Q02: *"extreme weather events caused by climate change"*** (MRR: `1.0000`, nDCG@10: `0.3037`):
   * **Causa del éxito**: Coincidencia exacta de bi-gramas y unigramas altamente discriminativos (*climate change*, *extreme weather*). En el corpus, los medios ambientales (*Climate Home News, Mongabay*) repiten textualmente estos términos, por lo que el ponderador TF-IDF asignó pesos elevados que posicionaron un documento altamente relevante en el **Rango #1 de inmediato**.
2. **Q13: *"coastal erosion and rising sea levels threatening settlements"*** (MRR: `1.0000`, nDCG@10: `0.2873`):
   * **Causa del éxito**: El término *erosion* y el bigrama *sea level* tienen un Document Frequency muy bajo en el corpus general, pero una concentración masiva en los artículos sobre atolones del Pacífico y manglares (*Tuvalu, Thailand coasts*). Esto generó una puntuación de coseno precisa que ubicó el documento idóneo en el primer lugar.
3. **Q08: *"renewable clean energy transition and solar power projects"*** (MRR: `0.3333`, P@10: `0.2000`):
   * **Causa del éxito**: Fuerte solapamiento léxico compuesto por múltiples palabras clave (*clean energy*, *solar*, *transition*). A pesar de que *energy* es común, la presencia conjunta de *solar power* elevó la precisión del Top-10.
4. **Q06: *"surging cost of food and grocery inflation"*** (MRR: `0.2500`, nDCG@10: `0.1158`):
   * **Causa del éxito**: La combinación de *inflation* con *food* y *cost* filtró notas económicas generales para seleccionar específicamente noticias sobre poder adquisitivo y supermercados en India y el Reino Unido.
5. **Q04: *"space exploration missions and astronomical discoveries"*** (MRR: `0.2500`, P@10: `0.1000`):
   * **Causa del éxito**: Términos con vocabulario técnico unívoco (*astronomical*, *exploration*, *space*). Al no presentar polisemia, los artículos de *Space.com* y *Nature Astronomy* emergieron rápidamente en los rangos superiores.

---

### B. Consultas con Bajo Rendimiento (Bottom 5) y Causas Observadas

1. **Q05: *"football match scores and championship tournament results"*** (P@10: `0.0000`, MRR: `0.0000`):
   * **Causa**: **Ausencia de solapamiento léxico por nombres propios de entidades**. Las noticias deportivas no dicen *"football match score"*, sino que usan nombres específicos de clubes y jugadores (*Arsenal vs Chelsea, Premier League, Lewandowski goal*). El modelo VSM es incapaz de saber que *"Arsenal"* es un equipo de *"football"* sin un embedding o base de conocimiento.
2. **Q07: *"digital cybersecurity attacks and data network breaches"*** (P@10: `0.0000`, MRR: `0.0000`):
   * **Causa**: **Sinonimia y jerga técnica no alineada**. Los artículos del corpus utilizan verbos y sustantivos coloquiales como *hacked, crypto wallet compromised, phishing, malware leak*, mientras que la consulta utilizó la terminología formal *cybersecurity attacks* y *data network breaches*. En VSM, si el término exacto no coincide, el producto punto es 0.
3. **Q09: *"clinical trials for novel cancer treatment therapies"*** (P@10: `0.0000`, MRR: `0.0000`):
   * **Causa**: **Fragmentación de vocabulario especializado**. Los artículos médicos del corpus discuten terapias específicas por su nombre biológico (*immunotherapy, oncology, chemotherapy, brown fat phenotyping, GLP-1 drug trials*). Sin expansión de consultas o representaciones densas, el término general *cancer treatment* no hace match con descripciones especializadas.
4. **Q10: *"electric vehicle market adoption and battery manufacturing"*** (P@10: `0.0000`, MRR: `0.0000`):
   * **Causa**: **Uso de siglas y términos de marca**. Los artículos describen *EVs, Waymo robotaxis, Tesla sales, lithium supply chains*. Al no haberse expandido la sigla *EV* a *electric vehicle* durante la normalización, el vector disperso no encontró intersección ortográfica.
5. **Q12: *"geopolitical conflicts disrupting global energy supplies"*** (P@10: `0.0000`, MRR: `0.0000`):
   * **Causa**: **Ambigüedad y distancia semántica profunda (Hard Query)**. La consulta describe una relación causal abstracta. Las noticias concretas hablan de *diesel prices, US-Russia deal, Palestine Action, Middle East oil tankers*. VSM carece de comprensión semántica composicional para inferir que una disputa en el estrecho de Ormuz representa un *geopolitical conflict*.

### Conclusión para la Etapa 2 (Dense Embeddings):
Este análisis demuestra experimentalmente el límite teórico del Modelo de Espacio Vectorial tradicional: **el problema del desajuste de vocabulario (vocabulary mismatch)**. Estos 5 casos de fallo representan la motivación perfecta para la **Etapa 2**, donde el uso de embeddings densos (ej. *Sentence-Transformers / BERT*) permitirá capturar similitudes semánticas latentes sin depender de la coincidencia ortográfica exacta.

---

## 5. Instrucciones de Verificación y Comandos

Todos los resultados reportados son 100% reproducibles mediante los siguientes comandos:

```bash
# 1. Normalización del corpus
python main.py normalize

# 2. Generación de modelos vectoriales (.pkl)
python main.py vectorize

# 3. Consulta individual en texto libre (Top-10)
python main.py search --query "artificial intelligence regulation" --content title --feature unigram --representation tfidf

# 4. Evaluación oficial del benchmark de 15 consultas
python main.py benchmark

# 5. Interfaz interactiva completa
python main.py interactive
```
