# Guía de estudio: noticias y K-Means, paso a paso

Elaboración conjunta de todos los integrantes del equipo: Ricardo Hernandez Espinoza; Diana Karen Espinoza Diaz; Byron Robles Lozano; Hector Arturo Saenz Buttanda; Diamanda Abigail Zurita Salinas.

Esta guía es para entender y exponer el trabajo. Primero explica las ideas, después qué hace cada archivo y finalmente dónde está cada requisito del profesor, con función, líneas, código exacto y lectura del fragmento. El informe de resultados sigue en resultados/INFORME_NO_SUPERVISADO.pdf.

Carpeta base: fase_2_aprendizaje/no_supervisada/. Las rutas de esta guía son relativas a esa carpeta. Los enlaces Markdown abren el código local. En el PDF se muestran números de línea y se dividen visualmente las líneas largas; para copiar código exacto usa Markdown o el archivo fuente.

## Cómo estudiar esta guía

- Primera lectura: sigue el recorrido y el glosario, sin intentar memorizar nombres de archivos.

- Segunda lectura: revisa las fichas de archivos para distinguir entradas, programas, modelo y resultados.

- Tercera lectura: estudia los apartados 1–23; en cada uno conecta requisito, idea, código y evidencia.

- Antes de exponer: practica las preguntas del final y usa inspeccionar.py para mostrar un grupo real.

- Para buscar algo concreto: usa Ctrl+F o Cmd+F con palabras como datos, validación, stopwords, silhouette, representantes o el nombre del TSV.

## El recorrido completo, contado sin código

1. Partimos de las mismas noticias del equipo. Revisamos IDs, texto y particiones, para no entrenar con un archivo desalineado o incompleto.

2. Cada candidato toma una forma de normalizar palabras, una representación y un rango de n-gramas. Esas decisiones producen una matriz de números.

3. Ajustamos las transformaciones con train; reducimos con SVD si corresponde, normalizamos L2 y entrenamos K-Means.

4. Representamos dev con esas mismas transformaciones y asignamos sus noticias con predict. Medimos separación y tamaños para comparar candidatos.

5. Elegimos KM083 mediante la regla declarada, revisamos sensibilidad a semillas y congelamos su configuración.

6. Reajustamos el pipeline completo en train+dev y guardamos el modelo final. Después asignamos test sin cambiar los parámetros.

7. Exportamos grupos, tamaños, palabras, noticias cercanas y lejanas, métricas y ejemplos de eventos. El equipo interpreta temas y limitaciones.

```text
Datos originales + texto + split
    -> normalización lingüística
    -> vectorizador (pesos + n-gramas)
    -> SVD opcional -> L2 -> K-Means
    -> comparación en dev -> selección
    -> reajuste con train+dev -> predict en test
    -> tablas + modelo + interpretación + informe
```

## Cuando dices revalidación: cuatro cosas diferentes

La palabra revalidar puede resultar ambigua. No hay que inventar un paso del PDF con ese nombre: las secciones 4, 5, 7 y 9 piden controles y evaluación concretos. Estas son las cuatro operaciones que sí puedes señalar en el proyecto:

Validar la integridad de los datos: Comprueba IDs, texto no vacío y coincidencia exacta con título + descripción. Es la revisión del archivo de entrada.

Ubicación precisa: experimentos.py, función load_documents, líneas 47–58. Los fragmentos completos se explican en los apartados correspondientes.

Validar candidatos con dev: Ajusta en train, representa dev, predice sus grupos y calcula silhouette y tamaños. Sirve para comparar configuraciones.

Ubicación precisa: experimentos.py, función run.trial, líneas 119–146. Los fragmentos completos se explican en los apartados correspondientes.

Reajustar el modelo final: final.fit(train_dev.text) aprende de nuevo con train+dev tras elegir. No es una comprobación del TSV.

Ubicación precisa: experimentos.py, función run, líneas 191–191. Los fragmentos completos se explican en los apartados correspondientes.

Auditar los resultados existentes: Contrasta matrices, métricas, representantes y particiones ya guardadas. No vuelve a seleccionar parámetros ni reentrena K-Means.

Ubicación precisa: auditar_cumplimiento.py, función run, líneas 20–104. Los fragmentos completos se explican en los apartados correspondientes.

## Diccionario para entender el código

Corpus: Colección de noticias. Aquí contiene 4645 registros; cada uno mantiene su ID.

Documento / noticia: Una unidad de entrada: título y descripción concatenados, no la URL ni la etiqueta.

Token y tokenización: Un token es una unidad obtenida al dividir texto. El vectorizador de esta corrida usa minúsculas y su patrón de tokens; no divide únicamente por espacios.

Stopword: Palabra incluida en una lista de términos de función comunes. Quitar esa lista no equivale a eliminar todo ruido editorial.

Lema y lematización: Forma básica de una palabra y proceso para obtenerla. Aquí spaCy también aporta categorías gramaticales que se filtran.

N-grama: Secuencia de n tokens consecutivos. Unigrama tiene uno; bigrama dos; trigrama tres.

Vocabulario: Conjunto ordenado de características aprendidas con los textos de ajuste. Una característica puede ser una palabra o un n-grama.

Matriz X: Tabla numérica de entrada al aprendizaje: una fila por noticia y una columna por característica o dimensión. Xtr corresponde a train y Xdv a dev.

Vector: Fila de números que representa una noticia en un espacio de características. La distancia se calcula entre estos vectores.

Representación binaria: Presencia/ausencia de un término: 1 si aparece, 0 si no aparece.

Representación de frecuencia: Número de apariciones del término en cada noticia.

TF-IDF: Ponderación que combina frecuencia en el documento con información sobre cuántos documentos contienen el término. Se aprende IDF solo con el conjunto de ajuste.

fit: Aprender parámetros a partir de datos: vocabulario, IDF, SVD o centros de K-Means, según el objeto.

transform: Convertir datos usando parámetros ya aprendidos. No es volver a ajustar el objeto.

fit_transform: Aprender una transformación y devolver la representación de los datos usados para aprenderla.

predict: Asignar cada vector al cluster aprendido más cercano. No crea una etiqueta de categoría ni actualiza los centros.

Pipeline: Secuencia de etapas. Evita aplicar transformaciones diferentes al entrenar y al usar el modelo.

Train / dev / test: Train aprende; dev compara candidatos; test aporta evidencia final tras elegir. Los tres tienen documentos separados durante selección.

Fuga de información: Utilizar información reservada para aprender o decidir. Ejemplo: ajustar el vocabulario con test o elegir parámetros por su puntuación test.

Hiperparámetro: Decisión que se configura antes de ajustar, como k o n_init. No es un centroide aprendido por K-Means.

Configuración y experimento: Una configuración reúne normalización, representación, n-gramas, SVD y parámetros. Un experimento es una ejecución identificada de esa configuración.

Cluster: Grupo de documentos asignados por cercanía. Su ID es arbitrario: cluster 2 no significa categoría 2.

Centroide: Vector que representa el centro de un grupo. Puede ser un promedio sin corresponder a ninguna noticia real.

k / n_clusters: Número de grupos solicitados al algoritmo. En el modelo elegido es 12.

n_init: Número de inicializaciones internas que K-Means prueba en cada ajuste. n_init=10 no significa diez clusters.

init: Regla para elegir los centros iniciales. Se comparó k-means++ y random.

max_iter: Límite de iteraciones por inicialización. El algoritmo puede converger antes; no es una orden de ejecutar siempre ese número.

Semilla / random_state: Control de la aleatoriedad para reproducir una ejecución en el mismo entorno. No elimina la necesidad de revisar sensibilidad a otras inicializaciones.

SVD: Reducción numérica de dimensiones. SVD=50 significa 50 coordenadas reducidas, no 50 palabras ni 50 clusters.

Normalización L2: Escalado de un vector no nulo a longitud uno. Es una operación numérica diferente de normalizar palabras.

Distancia euclidiana: Medida de separación entre vectores. Se utiliza tanto para asignar centros como para silhouette en esta corrida.

Silhouette: Medida de separación de los grupos, generalmente entre -1 y 1. No es un porcentaje de aciertos y no garantiza que el tema sea comprensible.

ARI: Medida de acuerdo entre dos particiones ajustada por azar. Aquí se usa de dos maneras: entre semillas para estabilidad y contra categorías para análisis externo.

NMI: Medida normalizada de información compartida entre particiones. En esta entrega compara clusters finales con categorías, sin entrenar con ellas.

Contingencia: Tabla que cuenta noticias por combinación de cluster y categoría. No es una matriz de confusión de un clasificador.

Reajuste / refit: Volver a aprender con la configuración ya escogida. Aquí el modelo final aprende con train+dev y conserva test reservado.

Vector cero: Noticia sin términos representados en el vocabulario usado. Se cuenta explícitamente; esta corrida no presentó vectores léxicos cero.

TSV / JSON / NPZ / joblib: TSV es una tabla de texto; JSON guarda estructuras y metadatos; NPZ guarda matrices; joblib serializa objetos de Python ya ajustados.

SHA-256 y manifiesto: Huella de bytes y lista de huellas de archivos para detectar cambios. Comprueban integridad, no calidad semántica.

## Tu captura: archivos anteriores y archivos actuales

La captura corresponde a la carpeta anterior fase_2_aprendizaje/agrupamiento/. Conservamos esa ejecución como referencia. La entrega nueva está en no_supervisada/. No mezcles sus métricas ni sus nombres de modelo: el trabajo anterior tenía 50 pruebas; el nuevo tiene 91 candidatos, tres ajustes de estabilidad y doce grupos finales.

Antes: ../agrupamiento/train_kmeans.py | Ahora: experimentos.py. El primero es el código anterior de 50 pruebas; el segundo es el proceso nuevo de 91 candidatos y tres ajustes de estabilidad.

Antes: ../agrupamiento/resultados/clustering_experiments.tsv | Ahora: resultados/experiments.tsv. 50 ensayos anteriores frente a 91 candidatos nuevos. No son la misma ejecución.

Antes: ../agrupamiento/resultados/clustering_results.tsv | Ahora: resultados/clustering_results.tsv. Comparten el nombre de salida, pero pertenecen a modelos diferentes; revisa siempre la carpeta.

Antes: ../agrupamiento/resultados/cluster_interpretation.md | Ahora: resultados/INFORME_NO_SUPERVISADO.md + top_words.tsv + representative_documents.tsv. La nueva explicación combina el informe con las tablas de términos y noticias.

Antes: ../agrupamiento/resultados/contingency_table.txt | Ahora: resultados/contingencia_train.tsv, contingencia_dev.tsv y contingencia_test.tsv. Ahora se separa el análisis externo por partición.

## Mapa de carpetas

```text
no_supervisada/
  datos/       Noticias originales, ampliación y texto con split
  soporte/     Normalizador del pipeline del equipo
  referencias/ PDF original del profesor
  modelos/     Pipeline ajustado y matrices binarias finales
  resultados/  Tablas, métricas, auditoría e informe
    figuras/   Gráficas incorporadas al informe
  *.py         Programas de aprendizaje, consulta y documentación
  *.json       Interpretación y material de estudio
  *.md / *.pdf Guía, instrucciones y estado
  *.zip        Paquete con toda la entrega
```

## Qué hace cada archivo y cómo se usa

Un archivo de resultados no contiene código ejecutable: en sus fichas explicamos qué significa y cómo se abre. El código que lo genera está identificado en los apartados por requisito. Una fila de TSV es un registro, no necesariamente una línea física si su texto incluye saltos. Para el código sí damos líneas físicas exactas.

### ESTADO_ENTREGA.md

Tipo: Resumen de cobertura. Qué hace: Relaciona entregables con evidencias y separa la rama K-Means del informe general de etapa II.

Cómo usarlo: Consultar como lista rápida, no como sustituto de esta guía explicativa.

### GUIA_DE_ESTUDIO_NO_SUPERVISADO.md

Tipo: Guía de estudio. Qué hace: Explica conceptos, archivos, requisitos y código exacto. Los nombres antiguos conservan el mismo contenido actualizado para no romper enlaces.

Cómo usarlo: Usar PDF para estudiar y Markdown para copiar código y seguir enlaces; no confundir con el informe de resultados.

### GUIA_DE_ESTUDIO_NO_SUPERVISADO.pdf

Tipo: Guía de estudio. Qué hace: Explica conceptos, archivos, requisitos y código exacto. Los nombres antiguos conservan el mismo contenido actualizado para no romper enlaces.

Cómo usarlo: Usar PDF para estudiar y Markdown para copiar código y seguir enlaces; no confundir con el informe de resultados.

### GUIA_REQUISITOS_Y_CODIGO.md

Tipo: Copia compatible de la guía. Qué hace: Explica conceptos, archivos, requisitos y código exacto. Los nombres antiguos conservan el mismo contenido actualizado para no romper enlaces.

Cómo usarlo: Usar PDF para estudiar y Markdown para copiar código y seguir enlaces; no confundir con el informe de resultados.

### GUIA_REQUISITOS_Y_CODIGO.pdf

Tipo: Copia compatible de la guía. Qué hace: Explica conceptos, archivos, requisitos y código exacto. Los nombres antiguos conservan el mismo contenido actualizado para no romper enlaces.

Cómo usarlo: Usar PDF para estudiar y Markdown para copiar código y seguir enlaces; no confundir con el informe de resultados.

### README.md

Tipo: Instrucciones de uso. Qué hace: Punto de entrada para preparar el entorno, consultar resultados y reproducir en una copia limpia. No entrena por abrirlo.

Cómo usarlo: Leer primero para ejecutar comandos correctamente.

### auditar_cumplimiento.py

Tipo: Auditoría profunda. Qué hace: Reconstruye el reparto; verifica las 2157 noticias originales; compara matrices completas, clusters, métricas, vocabulario y representantes. Guarda parámetros y 39 controles aprobados.

Cómo usarlo: No reentrena K-Means. Sí reconstruye un vectorizador para contrastar el vocabulario final y recalcula la evaluación ya guardada.

Funciones para localizar: run (líneas 20–104); run.check (líneas 23–26).

### contenido_estudio.json

Tipo: Explicaciones por requisito. Qué hace: Guarda explicaciones amplias, ejemplos, lectura del código y frases para exponer los 23 apartados.

Cómo usarlo: Leer o editar como texto; el generador lo incorpora a la guía.

### datos/labeled_documents.tsv

Tipo: Entrada con texto y particiones. Qué hace: Documentos con document_id, text, category y split, más metadatos. El mismo reparto se reutiliza en la tarea supervisada.

Cómo usarlo: Train/dev/test están dados en split; la categoría queda disponible solo para estratificación original y análisis externo.

### datos/news_raw_corpus_2157.tsv

Tipo: Corpus de etapa I. Qué hace: Copia de los 2157 registros originales que permite verificar preservación de IDs y contenido en la ampliación.

Cómo usarlo: Entrada de la auditoría; no es el corpus completo de la tarea B.

### datos/news_raw_corpus_4645.tsv

Tipo: Corpus ampliado original. Qué hace: 4645 noticias con título, descripción, fuente, categoría y metadatos. Es la referencia con la que se comprueba title + description.

Cómo usarlo: Entrada de load_documents(); no se vectorizan directamente URL ni category.

### ejecucion.log

Tipo: Registro de consola. Qué hace: Salida de la ejecución de aprendizaje, con IDs de experimentos, configuraciones, tamaños y finalización. Puede contener avisos no fatales del entorno.

Cómo usarlo: Buscar KM083 o Finalizado para localizar la decisión y la terminación de la corrida.

### entrega_no_supervisada.zip

Tipo: Paquete completo para compartir. Qué hace: Incluye código, datos, guías, informe, modelo y matrices locales. Evita perder binarios que el .gitignore excluye.

Cómo usarlo: Descomprimir para compartir la carpeta; no se ejecuta como programa.

### experimentos.py

Tipo: Programa principal de aprendizaje. Qué hace: Lee los datos; ejecuta las 91 comparaciones; elige con dev; revisa estabilidad; reajusta el pipeline final y exporta asignaciones, métricas, términos y representantes.

Cómo usarlo: Solo para entrenar una corrida nueva en una copia limpia. La actual está protegida contra sobrescritura.

Funciones para localizar: write_json (líneas 43–44); load_documents (líneas 47–58); feature_pipeline (líneas 61–68); kmeans (líneas 71–72); quality (líneas 75–79); lexical_summary (líneas 82–85); experiment_interpretation (líneas 88–96); run (líneas 99–236); run.trial (líneas 119–146); run.best (líneas 148–152); run.cfg_of (líneas 154–155).

### generar_guia.py

Tipo: Generación de esta guía. Qué hace: Extrae fragmentos reales mediante rangos y AST, incorpora explicaciones y fichas, verifica los códigos citados y genera las versiones de estudio. Consolida 91+3 ajustes.

Cómo usarlo: Si cambia el código fuente hay que regenerar la guía para actualizar las líneas.

Funciones para localizar: lines (líneas 19–20); span (líneas 23–27); function (líneas 30–34); consolidate_log (líneas 37–51); main (líneas 54–401); main.item (líneas 58–59); main.prose (líneas 173–175); main.code_block (líneas 176–183); main.source_context (líneas 184–195); main.explain_line (líneas 196–275); main.footer (líneas 388–389); main.StudyDocument.afterFlowable (líneas 391–394).

### generar_informe.py

Tipo: Generación de resultados escritos. Qué hace: Combina TSV, configuración e interpretación del equipo; dibuja las figuras y genera el informe analítico Markdown/PDF.

Cómo usarlo: Ejecutar para actualizar la presentación del informe, no para volver a ajustar clusters.

Funciones para localizar: fmt (líneas 21–22); generate (líneas 25–182); generate.add (líneas 67–67); generate.footer (líneas 177–180).

### inspeccionar.py

Tipo: Consulta por consola. Qué hace: Lee TSV para mostrar métricas/grupos y carga el joblib al usar --texto. No actualiza el modelo.

Cómo usarlo: Usar para demostrar resultados: sin argumentos, con --cluster N o con --texto.

### interpretacion_manual.json

Tipo: Lectura temática del equipo. Qué hace: Propone nombre y análisis de cada cluster, comenta estabilidad y documenta pares de eventos. No contiene respuestas objetivo para entrenar.

Cómo usarlo: Revisar si se hace una corrida nueva: los números de cluster pueden cambiar de significado.

### material_apoyo_estudio.py

Tipo: Material didáctico. Qué hace: Contiene glosario, fichas de archivos, equivalencias con la captura anterior y preguntas de repaso. No participa en el entrenamiento.

Cómo usarlo: Editar para mejorar explicaciones; no modifica las asignaciones.

### modelos/X_words_dev.npz

Tipo: Matriz léxica final de dev. Qué hace: Representación de las 697 noticias dev con el vocabulario final. Dev ya participó en el reajuste de ese modelo.

Cómo usarlo: No confundir con las matrices dev de cada candidato durante selección.

### modelos/X_words_test.npz

Tipo: Matriz léxica final de test. Qué hace: Representación de las 697 noticias reservadas, usando el vocabulario aprendido en train+dev.

Cómo usarlo: No se utiliza para escoger parámetros. Su orden coincide con el TSV filtrado por test.

### modelos/X_words_train.npz

Tipo: Matriz léxica final de train. Qué hace: Representación de las 3251 noticias train con el vocabulario del reajuste final. Todavía está en columnas de términos, antes de SVD y L2.

Cómo usarlo: Usar scipy.sparse.load_npz. El orden de documentos es el TSV filtrado por train.

### modelos/kmeans_pipeline.joblib

Tipo: Modelo final ajustado. Qué hace: Objeto con normalizador, vectorizador, SVD=50, L2 y K-Means k=12 ya entrenados con 3948 textos.

Cómo usarlo: Se abre con joblib.load dentro de los scripts que preparan la importación de NormalizadorNoticias. No es un archivo legible como texto.

### referencias/Proyecto_01_Noticias_Etapa_II.pdf

Tipo: Documento del profesor. Qué hace: Fuente de los requisitos: corpus y protocolo, representaciones, tarea B, bitácora, interfaz e informe.

Cómo usarlo: Es referencia para estudiar los requisitos, no un script.

### requirements-lock.txt

Tipo: Dependencias exactas. Qué hace: Versiones de bibliotecas y URL con huella del modelo inglés de spaCy. Es la receta del entorno utilizado, no un resultado del modelo.

Cómo usarlo: Se instala con pip -r; la instalación necesita acceso a las fuentes de paquetes.

### resultados/INFORME_NO_SUPERVISADO.md

Tipo: Informe editable. Qué hace: Metodología, resultados, análisis y limitaciones, con autoría conjunta.

Cómo usarlo: Versión de texto del informe científico, distinta de esta guía de estudio.

### resultados/INFORME_NO_SUPERVISADO.pdf

Tipo: Informe de resultados. Qué hace: Versión de presentación con gráficas y evidencia real.

Cómo usarlo: Usar para entregar el análisis de la rama K-Means e integrarlo con el informe general.

### resultados/auditoria_cumplimiento.json

Tipo: 39 comprobaciones detalladas. Qué hace: Lista los controles técnicos con estado y evidencia; incluye alcance y autoría conjunta.

Cómo usarlo: Leer checks para ver qué se verificó realmente, no asumir que PASS equivale a temas perfectos.

### resultados/cluster_sizes.tsv

Tipo: Tamaños por partición. Qué hace: Doce filas, una por grupo, con train/dev/test y total.

Cómo usarlo: Sumar total debe dar 4645; el cluster 0 totaliza 951.

### resultados/clustering_results.tsv

Tipo: Asignación de cada noticia. Qué hace: 4645 filas con document_id, split, cluster_id y method/configuration.

Cómo usarlo: Buscar un ID concreto y después ir al texto del documento con ese mismo ID.

### resultados/comparaciones_controladas.tsv

Tipo: Parejas de un factor. Qué hace: Dos candidatos con los demás factores constantes y delta de silhouette B-A.

Cómo usarlo: Un delta positivo favorece B en esa pareja, no prueba una mejora universal.

### resultados/configuracion_final.json

Tipo: Metadatos del modelo entregado. Qué hace: Guarda configuración ganadora, 3948 textos de ajuste, versiones, semilla y test_used_for_selection=False.

Cómo usarlo: Describe el objeto final; no contiene por sí mismo los pesos y centros, que están en joblib.

### resultados/contingencia_dev.tsv

Tipo: Tabla externa de categorías. Qué hace: Recuento de categorías RSS por cluster para dev. No son entradas de aprendizaje ni una matriz de confusión supervisada.

Cómo usarlo: Una fila es un cluster y una columna una categoría; cada celda cuenta documentos.

### resultados/contingencia_test.tsv

Tipo: Tabla externa de categorías. Qué hace: Recuento de categorías RSS por cluster para test. No son entradas de aprendizaje ni una matriz de confusión supervisada.

Cómo usarlo: Una fila es un cluster y una columna una categoría; cada celda cuenta documentos.

### resultados/contingencia_train.tsv

Tipo: Tabla externa de categorías. Qué hace: Recuento de categorías RSS por cluster para train. No son entradas de aprendizaje ni una matriz de confusión supervisada.

Cómo usarlo: Una fila es un cluster y una columna una categoría; cada celda cuenta documentos.

### resultados/corpus_normalizado_final.tsv

Tipo: Textos después de normalizar. Qué hace: ID, split y text_normalized de cada documento con el modo final sin_stopwords.

Cómo usarlo: Permite comprobar qué entró al vectorizador; no es la matriz numérica ni el texto original completo.

### resultados/dev_cluster_interpretation.tsv

Tipo: Evidencia dev de todos los candidatos. Qué hace: Por experimento y cluster guarda tamaño dev, diez términos y hasta dos textos representativos.

Cómo usarlo: Permite inspeccionar configuraciones que no ganaron, incluidos grupos extremos.

### resultados/distribucion_corpus.tsv

Tipo: Recuento por categoría y split. Qué hace: Comprueba que las diez categorías tengan ejemplos en las tres particiones. No define los clusters.

Cómo usarlo: Abrir como tabla y comparar train/dev/test por fila.

### resultados/estabilidad_dev.json

Tipo: Sensibilidad a semillas. Qué hace: Tres silhouttes y tamaños dev por semilla, y tres ARI entre pares de asignaciones.

Cómo usarlo: ARI aquí compara ejecuciones, no las categorías RSS.

### resultados/eventos_recurrentes.tsv

Tipo: Pares de coberturas del mismo evento. Qué hace: Tres casos con IDs, grupos, same_cluster e interpretación.

Cómo usarlo: Leer los dos documentos para entender tanto ejemplos favorables como separaciones.

### resultados/experiments.tsv

Tipo: Bitácora de 91 candidatos. Qué hace: Cada fila identifica configuración y resultado dev. Incluye tamaños, elegibilidad, convergencia y selected.

Cómo usarlo: Filtrar selected=True para localizar KM083; eligible=False identifica candidatos rechazados.

### resultados/experiments_completo.tsv

Tipo: Bitácora consolidada de 94 ajustes. Qué hace: Añade tres ejecuciones de estabilidad a las 91 de selección. candidate_for_selection separa las dos finalidades.

Cómo usarlo: Las semillas de estabilidad no se comparan como candidatas para cambiar el ganador.

### resultados/figuras/experimentos_dev.png

Tipo: Gráfica de candidatos. Qué hace: Silhouette dev por experimento, diferenciando admisibles y elegido.

Cómo usarlo: Muestra la comparación; no incorpora test para elegir.

### resultados/figuras/tamanos_clusters.png

Tipo: Gráfica de tamaños. Qué hace: Barras apiladas de train/dev/test para los doce grupos.

Cómo usarlo: Apoya la explicación de grupos grandes y pequeños.

### resultados/labeled_documents.tsv

Tipo: Instantánea de datos utilizados. Qué hace: Copia exportada de la entrada para acompañar las asignaciones y permitir rastrear IDs, textos y splits.

Cómo usarlo: La auditoría comprueba igualdad con datos/labeled_documents.tsv.

### resultados/manifest.json

Tipo: Integridad del paquete. Qué hace: Ruta, tamaño y SHA-256 de los archivos entregados, excepto el propio manifiesto y el ZIP.

Cómo usarlo: Se actualiza al empaquetar para que no conserve huellas de una guía anterior.

### resultados/metricas_finales.tsv

Tipo: Evaluación del objeto reajustado. Qué hace: Tres filas: train, dev y test. Incluye silhouette, espacio, distancia, tamaños, ARI/NMI externos y vectores cero.

Cómo usarlo: El dev de esta tabla ya entró al ajuste: no es la estimación independiente que seleccionó KM083.

### resultados/parametros_pipeline_final.json

Tipo: Parámetros completos. Qué hace: Incluye los parámetros de cada etapa del modelo cargado, también opciones por defecto del vectorizador.

Cómo usarlo: Complementa la configuración resumida para responder preguntas de implementación.

### resultados/plan_experimental.json

Tipo: Plan del experimento. Qué hace: Registra variantes, controles del vectorizador, etapas, semilla, criterio de admisibilidad y huella del corpus etiquetado.

Cómo usarlo: Explica la búsqueda antes de mirar el archivo del ganador.

### resultados/representative_documents.tsv

Tipo: Noticias para interpretar grupos. Qué hace: 36 filas: dos representantes y una frontera por grupo. Guarda texto, URL, distancia, ID y partición.

Cómo usarlo: role=representativo son las 24 noticias pedidas; role=frontera son doce ejemplos adicionales.

### resultados/resumen_factores.tsv

Tipo: Resumen descriptivo del factorial. Qué hace: Medias de silhouette y número de admisibles por valor de factor.

Cómo usarlo: Es una visión global; las parejas controladas son mejores para explicar un cambio aislado.

### resultados/seleccion_dev.json

Tipo: Decisión antes del reajuste. Qué hace: Identifica KM083, sus parámetros, regla de selección y silhouette dev 0.165800.

Cómo usarlo: Es el archivo correcto para explicar por qué ganó; no usar dev del modelo final para justificarlo.

### resultados/top_words.tsv

Tipo: Términos de los centros. Qué hace: 120 filas: doce grupos por diez posiciones. Los términos se obtuvieron en el espacio original de palabras.

Cómo usarlo: Filtrar cluster_id y ordenar rank para leer los diez términos de ese grupo.

### resultados/validacion.txt

Tipo: Resultado de comprobaciones breves. Qué hace: Mensaje OK de validar_resultados.py con conteos y consistencia comprobada.

Cómo usarlo: Es evidencia de ejecución del validador, no una métrica semántica.

### resultados/vocabulario.tsv

Tipo: Nombres de columnas léxicas. Qué hace: Un término por fila en el mismo orden que las columnas de las matrices NPZ.

Cómo usarlo: No es una lista exclusiva de palabras representativas; contiene todo el vocabulario final.

### soporte/normalizacion.py

Tipo: Normalizador reutilizado. Qué hace: Copia del módulo del pipeline del equipo. Define limpieza, carga de spaCy, filtrado de lemas y clase NormalizadorNoticias.

Cómo usarlo: Se importa al entrenar y al cargar el joblib, para que la clase serializada pueda encontrarse.

Funciones para localizar: limpiar_texto (líneas 30–35); cargar_spacy (líneas 39–40); lematizar (líneas 43–51); NormalizadorNoticias.__init__ (líneas 57–58); NormalizadorNoticias.fit (líneas 60–61); NormalizadorNoticias.transform (líneas 63–74).

### validar_resultados.py

Tipo: Comprobaciones de integridad. Qué hace: Comprueba correspondencia de IDs, sumas, selección, representantes y persistencia del modelo. Es más breve que la auditoría profunda.

Cómo usarlo: Ejecutar para verificar consistencia de las salidas existentes.

Funciones para localizar: main (líneas 17–63).

## Requisitos del profesor explicados y localizados

Cada referencia al PDF es una explicación fiel en español, no una cita literal. Primero se explica la intención, después la implementación y finalmente el código y la evidencia. El número del apartado de esta guía no es el número de sección del PDF: se indica por separado.

## 1. Reutilizar el corpus, IDs y título + descripción

Qué pide el profesor: consultar sección(es) 4 y 11 del PDF. Resumen del requisito: Se leen las noticias originales y se comprueba que cada texto sea exactamente título + espacio + descripción. Las etiquetas se conservan como metadatos externos. Se comprobó además que los 2157 registros de etapa I permanecen idénticos en el corpus ampliado. No se hace scraping nuevo.

### Para entenderlo desde cero

Un corpus es la colección de noticias que utilizamos como datos. Aquí cada noticia conserva un identificador, por ejemplo N2256. Ese ID permite seguir el mismo documento desde el archivo original hasta el cluster asignado: el modelo trabaja con números, pero nosotros podemos regresar al texto gracias al ID.

El profesor establece que la entrada sea título + descripción. Por eso no entrenamos con la URL, el nombre de la fuente ni la categoría. load_documents() lee dos TSV y los cruza por identificador. El cruce uno a uno evita que un mismo ID produzca varias filas por un duplicado. Después assert comprueba que el texto y la categoría correspondan al original.

Hay dos revisiones diferentes: la lectura del programa comprueba integridad de las 4645 noticias; la auditoría también compara las primeras 2157 noticias de etapa I con el corpus ampliado para verificar que no se sustituyeron sus registros. Esta validación detecta errores de los datos: todavía no mide qué tan buenos son los clusters.

Ejemplo: Ejemplo ilustrativo: título="New cancer study" y descripción="Researchers test a treatment" producen "New cancer study Researchers test a treatment". La URL se conserva para consultar la noticia, pero no se concatena al texto del modelo.

### Dónde está y qué archivo demuestra el resultado

- datos/news_raw_corpus_2157.tsv: Copia de los 2157 registros originales que permite verificar preservación de IDs y contenido en la ampliación.

- datos/news_raw_corpus_4645.tsv: 4645 noticias con título, descripción, fuente, categoría y metadatos. Es la referencia con la que se comprueba title + description.

- datos/labeled_documents.tsv: Documentos con document_id, text, category y split, más metadatos. El mismo reparto se reutiliza en la tarea supervisada.

- resultados/labeled_documents.tsv: Copia exportada de la entrada para acompañar las asignaciones y permitir rastrear IDs, textos y splits.

### Cómo leer la implementación

- pd.read_csv(..., sep="\t") lee una tabla separada por tabuladores.

- document_id.is_unique comprueba que los IDs no se repitan.

- merge(..., validate="one_to_one") exige una noticia por ID a ambos lados.

- assert compara el texto ya guardado con title + espacio + description.

- Si un assert falla, el programa se detiene; no inventa datos ni continúa ocultando el error.

Archivo y enlace exacto: [experimentos.py](/Users/diamandazurita/Downloads/NLP_News_Project/fase_2_aprendizaje/no_supervisada/experimentos.py:47). Función o contexto: `load_documents`. Líneas 47–58.

```python
def load_documents():
    path = HERE / 'datos/labeled_documents.tsv'
    docs = pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False)
    assert docs.document_id.is_unique and set(docs.split) == {'train', 'dev', 'test'}
    assert docs.text.str.strip().ne('').all() and docs.category.ne('').all()
    # Comprueba que texto y etiquetas corresponden al corpus original, sin cambiar los splits.
    raw = pd.read_csv(HERE / 'datos/news_raw_corpus_4645.tsv', sep='\t', dtype=str, keep_default_na=False)
    merged = docs.merge(raw, left_on='document_id', right_on='id', suffixes=('_labeled', '_raw'), validate='one_to_one')
    assert len(merged) == len(docs)
    assert (merged.text == (merged.title + ' ' + merged.description).str.strip()).all()
    assert (merged.category_labeled == merged.category_raw).all()
    return docs, path
```

Lectura de las líneas de este fragmento:

- Línea 47: Define la función que se invoca desde otro punto del programa; definirla todavía no ejecuta su contenido.

- Línea 48: Asigna el valor o resultado de la derecha a la variable de la izquierda para usarlo en los pasos siguientes.

- Línea 49: Lee una tabla del disco; sep=\t indica que sus columnas están separadas por tabuladores.

- Línea 50: Exige que la condición sea verdadera. Si no lo es, la ejecución se detiene con AssertionError.

- Línea 51: Exige que la condición sea verdadera. Si no lo es, la ejecución se detiene con AssertionError.

- Línea 52: Comentario que documenta la intención; Python no lo ejecuta.

- Línea 53: Lee una tabla del disco; sep=\t indica que sus columnas están separadas por tabuladores.

- Línea 54: Cruza documentos por ID y exige que cada ID aparezca una sola vez en ambos lados.

- Línea 55: Exige que la condición sea verdadera. Si no lo es, la ejecución se detiene con AssertionError.

- Línea 56: Exige que la condición sea verdadera. Si no lo es, la ejecución se detiene con AssertionError.

- Línea 57: Exige que la condición sea verdadera. Si no lo es, la ejecución se detiene con AssertionError.

- Línea 58: Devuelve el resultado de la función a quien la llamó; no implica que se guarde automáticamente en disco.

Archivo y enlace exacto: [auditar_cumplimiento.py](/Users/diamandazurita/Downloads/NLP_News_Project/fase_2_aprendizaje/no_supervisada/auditar_cumplimiento.py:29). Función o contexto: `run`. Líneas 29–31.

```python
    original=pd.read_csv(HERE/'datos/news_raw_corpus_2157.tsv',sep='\t',dtype=str,keep_default_na=False)
    preserved=raw.set_index('id').reindex(original.id).reset_index()
    check('IDs y artículos de etapa I preservados',original.equals(preserved[original.columns]),'Los 2157 registros originales permanecen idénticos en el corpus ampliado de 4645')
```

Lectura de las líneas de este fragmento:

- Línea 29: Lee una tabla del disco; sep=\t indica que sus columnas están separadas por tabuladores.

- Línea 30: Asigna el valor o resultado de la derecha a la variable de la izquierda para usarlo en los pasos siguientes.

- Línea 31: Continúa la expresión de la línea anterior; hay que leerla dentro de la llamada o estructura completa mostrada arriba.

Para explicarlo con tus palabras: Primero comprobamos que cada noticia conserve su ID y que la entrada sea exactamente título más descripción; así sabemos qué documento representa cada fila.

## 2. Particiones train/dev/test reproducibles

Qué pide el profesor: consultar sección(es) 4 del PDF. Resumen del requisito: El entrenamiento reutiliza los splits existentes: 3251/697/697. La auditoría reproduce de forma independiente el doble train_test_split, con shuffle=True, semilla 0 y estratificación, y comprueba que coincida noticia por noticia. La estratificación original usa categorías para repartir, no para entrenar K-Means.

### Para entenderlo desde cero

Train, dev y test son tres conjuntos de documentos distintos. Train es el material con el que se aprende. Dev sirve para comparar configuraciones sin entrenarlas con esas noticias. Test se reserva para informar el resultado final una vez que la elección ya está cerrada.

Nuestro módulo no vuelve a repartir el corpus al entrenar: toma la columna split que ya existe y que comparte con la supervisada. La auditoría reconstruye el reparto por separado para comprobarlo. Primero deja 70% en train y 30% restante; después divide ese 30% por mitad. Se obtienen 3251, 697 y 697 noticias, respectivamente.

shuffle=True mezcla las filas; random_state=0 hace reproducible la mezcla. stratify conserva aproximadamente la proporción de categorías. Usar la categoría para estratificar el reparto no equivale a enseñarle la categoría a K-Means: el ajuste posterior recibe únicamente los vectores de texto. No basta con que los tamaños coincidan: también se comprueba qué IDs quedaron en cada conjunto.

Ejemplo: Piensa en preparar un examen: train es el material de práctica, dev un examen de ensayo para comparar estrategias y test la evaluación final que no utilizas para escoger la estrategia.

### Dónde está y qué archivo demuestra el resultado

- resultados/distribucion_corpus.tsv: Comprueba que las diez categorías tengan ejemplos en las tres particiones. No define los clusters.

- resultados/auditoria_cumplimiento.json: Lista los controles técnicos con estado y evidencia; incluye alcance y autoría conjunta.

### Cómo leer la implementación

- docs[docs.split == "train"] selecciona únicamente filas de entrenamiento.

- reset_index(drop=True) renumera las posiciones locales sin cambiar document_id.

- train_test_split(... test_size=.30) aparta el 30% restante.

- La segunda división usa test_size=.50 sobre ese restante.

- check compara los nombres de partición reconstruidos con los splits reales.

Archivo y enlace exacto: [experimentos.py](/Users/diamandazurita/Downloads/NLP_News_Project/fase_2_aprendizaje/no_supervisada/experimentos.py:106). Función o contexto: `run`. Líneas 106–107.

```python
    train = docs[docs.split == 'train'].reset_index(drop=True)
    dev = docs[docs.split == 'dev'].reset_index(drop=True)
```

Lectura de las líneas de este fragmento:

- Línea 106: Filtra, conserva o comprueba la partición de los documentos; no la convierte en característica del texto.

- Línea 107: Filtra, conserva o comprueba la partición de los documentos; no la convierte en característica del texto.

Archivo y enlace exacto: [auditar_cumplimiento.py](/Users/diamandazurita/Downloads/NLP_News_Project/fase_2_aprendizaje/no_supervisada/auditar_cumplimiento.py:35). Función o contexto: `run`. Líneas 35–38.

```python
    train,rest=train_test_split(docs.index,test_size=.30,shuffle=True,random_state=0,stratify=docs.category)
    dev,test=train_test_split(rest,test_size=.50,shuffle=True,random_state=0,stratify=docs.loc[rest,'category'])
    check('Particiones reproducibles 70/15/15',all(docs.loc[idx,'split'].eq(name).all() for name,idx in [('train',train),('dev',dev),('test',test)]),'Reconstrucción independiente con train_test_split, shuffle=True, random_state=0 y estratificación')
    check('Todas las clases en cada partición',pd.crosstab(docs.category,docs.split).gt(0).all().all(),'Diez categorías presentes en train, dev y test')
```

Lectura de las líneas de este fragmento:

- Línea 35: Realiza el reparto reproducible que la auditoría compara con los splits existentes.

- Línea 36: Realiza el reparto reproducible que la auditoría compara con los splits existentes.

- Línea 37: Realiza el reparto reproducible que la auditoría compara con los splits existentes.

- Línea 38: Cruza dos variables y cuenta cuántos documentos hay en cada combinación.

Para explicarlo con tus palabras: Conservamos el reparto 70/15/15 del equipo y comprobamos que sea reproducible con semilla 0, sin mezclar documentos entre conjuntos.

## 3. Excluir las categorías del entrenamiento y aislar dev

Qué pide el profesor: consultar sección(es) 4, 5 y 7 del PDF. Resumen del requisito: Las matrices se aprenden con textos train; dev solo se transforma. fit(Xtr) recibe características, sin category ni un vector y. Las categorías se usan únicamente al calcular métricas externas y contingencia después de congelar la configuración.

### Para entenderlo desde cero

No supervisado significa que el algoritmo no recibe una respuesta correcta para cada noticia durante el ajuste. En clasificación se entrenaría con texto y categoría. Aquí K-Means recibe solamente Xtr, la matriz numérica de train; no recibe una columna category ni un vector y de etiquetas.

Las filas representan noticias y las columnas características de texto. El número que devuelve K-Means, por ejemplo cluster 2, es un identificador interno del grupo; no significa automáticamente política ni coincide por obligación con una categoría del RSS. Nosotros proponemos un nombre leyendo palabras y noticias después.

fit_transform(tr) aprende el vocabulario y, si corresponde, el IDF y la reducción SVD con train. transform(dv) utiliza esos objetos ya aprendidos para representar dev. Si se hiciera fit_transform(dv), dev modificaría la representación y la comparación dejaría de estar aislada: eso sería fuga de información.

Ejemplo: En esta corrida N2256 queda en cluster 2. El número 2 no fue proporcionado por el profesor como etiqueta; surgió del agrupamiento. El tema de política estadounidense se interpreta después.

### Dónde está y qué archivo demuestra el resultado

- resultados/experiments.tsv: Cada fila identifica configuración y resultado dev. Incluye tamaños, elegibilidad, convergencia y selected.

- resultados/seleccion_dev.json: Identifica KM083, sus parámetros, regla de selección y silhouette dev 0.165800.

### Cómo leer la implementación

- normalized[modo] contiene dos listas de texto: train y dev.

- fp.fit_transform(tr) aprende las transformaciones con train.

- fp.transform(dv) aplica las mismas transformaciones a dev.

- kmeans(cfg).fit(Xtr) ajusta los centros sin pasar categorías.

- km.predict(Xdv) asigna dev a los centros ya aprendidos.

Archivo y enlace exacto: [experimentos.py](/Users/diamandazurita/Downloads/NLP_News_Project/fase_2_aprendizaje/no_supervisada/experimentos.py:128). Función o contexto: `run.trial`. Líneas 128–130.

```python
            tr, dv = normalized[cfg['normalization']]
            Xtr = fp.fit_transform(tr)
            Xdv = fp.transform(dv)
```

Lectura de las líneas de este fragmento:

- Línea 128: Asigna el valor o resultado de la derecha a la variable de la izquierda para usarlo en los pasos siguientes.

- Línea 129: Aprende la representación exclusivamente con textos train y devuelve su matriz.

- Línea 130: Representa dev con los objetos ya ajustados; aquí no se aprende vocabulario nuevo.

Archivo y enlace exacto: [experimentos.py](/Users/diamandazurita/Downloads/NLP_News_Project/fase_2_aprendizaje/no_supervisada/experimentos.py:135). Función o contexto: `run.trial`. Líneas 135–138.

```python
        fp, Xtr, Xdv = caches[featurekey]
        km = kmeans(cfg).fit(Xtr)
        labels = km.predict(Xdv)
        q = quality(Xdv, labels, cfg['k'])
```

Lectura de las líneas de este fragmento:

- Línea 135: Reutiliza las mismas matrices y transformaciones para esa configuración de características, sin incorporar textos nuevos.

- Línea 136: Ajusta K-Means con la matriz de entrenamiento, sin pasar categorías como respuesta.

- Línea 137: Asigna los documentos dev a los centros aprendidos con train.

- Línea 138: Asigna el valor o resultado de la derecha a la variable de la izquierda para usarlo en los pasos siguientes.

Para explicarlo con tus palabras: Las etiquetas siguen en el archivo para análisis externo, pero no se entregan al entrenamiento ni al selector de configuraciones.

## 4. Tres normalizaciones, incluida una mínima

Qué pide el profesor: consultar sección(es) 5 y 7 del PDF. Resumen del requisito: minima deja el texto a la tokenización y minúsculas del vectorizador. sin_stopwords tokeniza y filtra la lista inglesa. lemas limpia boilerplate, aplica spaCy en inglés y filtra POS. Esta última no equivale a aplicar exactamente la misma lista de stopwords. El archivo de soporte es una copia del normalizador del equipo.

### Para entenderlo desde cero

La normalización lingüística modifica el texto antes de contar términos. La comparación necesita una base mínima para saber si los pasos adicionales realmente ayudan. minima devuelve el texto tal como llegó a esta etapa; luego el vectorizador lo convierte a minúsculas y tokeniza.

sin_stopwords tokeniza y quita palabras de la lista inglesa de scikit-learn, como artículos y conectores frecuentes. No quita automáticamente frases editoriales como Continue reading. lemas primero limpia algunas fórmulas de texto y después usa spaCy para obtener lemas y filtrar ciertas clases gramaticales, como pronombres y auxiliares.

Estas variantes no son idénticas salvo por una sola operación: lemas combina limpieza, lematización y filtrado POS. Por eso hay que describirla completa y no decir que cualquier diferencia se debe únicamente a lematizar. La transformación es fija, no aprende frecuencias del conjunto completo; el aprendizaje del vocabulario ocurre después en train.

Ejemplo: Ejemplo ilustrativo: una variante puede dejar "the researchers are studying", otra filtrar palabras de función y otra llevar "researchers" a "researcher". No usamos estos ejemplos como resultados medidos del corpus.

### Dónde está y qué archivo demuestra el resultado

- soporte/normalizacion.py: Copia del módulo del pipeline del equipo. Define limpieza, carga de spaCy, filtrado de lemas y clase NormalizadorNoticias.

- resultados/experiments.tsv: Cada fila identifica configuración y resultado dev. Incluye tamaños, elegibilidad, convergencia y selected.

- resultados/corpus_normalizado_final.tsv: ID, split y text_normalized de cada documento con el modo final sin_stopwords.

### Cómo leer la implementación

- self.modo decide qué rama de transform() se aplica.

- minima devuelve list(X).

- sin_stopwords usa tokenizar y ENGLISH_STOP_WORDS.

- limpiar_texto aplica las reglas de limpieza de boilerplate.

- cargar_spacy().pipe procesa textos por lotes; lematizar revisa token.lemma_ y token.pos_.

Archivo y enlace exacto: [experimentos.py](/Users/diamandazurita/Downloads/NLP_News_Project/fase_2_aprendizaje/no_supervisada/experimentos.py:35). Función o contexto: `nivel principal del módulo`. Líneas 35–37.

```python
NORMS = ['minima', 'sin_stopwords', 'lemas']
REPS = ['binaria', 'frecuencia', 'tfidf']
GRAMS = [(1, 1), (1, 2), (1, 3)]
```

Lectura de las líneas de este fragmento:

- Línea 35: Asigna el valor o resultado de la derecha a la variable de la izquierda para usarlo en los pasos siguientes.

- Línea 36: Asigna el valor o resultado de la derecha a la variable de la izquierda para usarlo en los pasos siguientes.

- Línea 37: Asigna el valor o resultado de la derecha a la variable de la izquierda para usarlo en los pasos siguientes.

Archivo y enlace exacto: [soporte/normalizacion.py](/Users/diamandazurita/Downloads/NLP_News_Project/fase_2_aprendizaje/no_supervisada/soporte/normalizacion.py:63). Función o contexto: `NormalizadorNoticias.transform`. Líneas 63–74.

```python
    def transform(self, X):
        if self.modo == "minima":
            return list(X)
        if self.modo == "sin_stopwords":
            return [
                " ".join(token for token in tokenizar(texto) if token not in ENGLISH_STOP_WORDS)
                for texto in X
            ]
        textos = [limpiar_texto(texto) for texto in X]
        if self.modo == "limpieza":
            return textos
        return [lematizar(doc) for doc in cargar_spacy().pipe(textos, batch_size=256)]
```

Lectura de las líneas de este fragmento:

- Línea 63: Define la función que se invoca desde otro punto del programa; definirla todavía no ejecuta su contenido.

- Línea 64: Evalúa la condición y ejecuta esta rama únicamente cuando corresponde.

- Línea 65: Devuelve el resultado de la función a quien la llamó; no implica que se guarde automáticamente en disco.

- Línea 66: Evalúa la condición y ejecuta esta rama únicamente cuando corresponde.

- Línea 67: Devuelve el resultado de la función a quien la llamó; no implica que se guarde automáticamente en disco.

- Línea 68: Elimina tokens pertenecientes a la lista inglesa; no elimina todo tipo de boilerplate.

- Línea 69: Repite el bloque para cada valor de la colección indicada, por ejemplo una configuración, semilla o grupo.

- Línea 70: Continúa la expresión de la línea anterior; hay que leerla dentro de la llamada o estructura completa mostrada arriba.

- Línea 71: Asigna el valor o resultado de la derecha a la variable de la izquierda para usarlo en los pasos siguientes.

- Línea 72: Evalúa la condición y ejecuta esta rama únicamente cuando corresponde.

- Línea 73: Devuelve el resultado de la función a quien la llamó; no implica que se guarde automáticamente en disco.

- Línea 74: Procesa varios textos con spaCy por lotes, en vez de tratar cada texto como una llamada independiente.

Archivo y enlace exacto: [soporte/normalizacion.py](/Users/diamandazurita/Downloads/NLP_News_Project/fase_2_aprendizaje/no_supervisada/soporte/normalizacion.py:43). Función o contexto: `lematizar`. Líneas 43–51.

```python
def lematizar(doc):
    lemas = []
    for token in doc:
        if token.pos_ in POS_DESCARTADOS or token.is_punct or token.is_space:
            continue
        lema = re.sub(r"^[^\w]+|[^\w]+$", "", token.lemma_.lower().strip())
        if lema:
            lemas.append(lema)
    return " ".join(lemas)
```

Lectura de las líneas de este fragmento:

- Línea 43: Define la función que se invoca desde otro punto del programa; definirla todavía no ejecuta su contenido.

- Línea 44: Asigna el valor o resultado de la derecha a la variable de la izquierda para usarlo en los pasos siguientes.

- Línea 45: Repite el bloque para cada valor de la colección indicada, por ejemplo una configuración, semilla o grupo.

- Línea 46: Consulta su clase gramatical para decidir si se descarta en esta variante.

- Línea 47: Continúa la expresión de la línea anterior; hay que leerla dentro de la llamada o estructura completa mostrada arriba.

- Línea 48: Obtiene el lema propuesto por spaCy para ese token.

- Línea 49: Evalúa la condición y ejecuta esta rama únicamente cuando corresponde.

- Línea 50: Agrega un elemento a la colección que se exportará o presentará después.

- Línea 51: Devuelve el resultado de la función a quien la llamó; no implica que se guarde automáticamente en disco.

Para explicarlo con tus palabras: Probamos tres niveles de procesamiento: una base mínima, quitar stopwords y una variante con limpieza y lemas; explicamos exactamente qué cambia en cada una.

## 5. Binaria, frecuencia, TF-IDF y rangos de n-gramas

Qué pide el profesor: consultar sección(es) 5 y 7 del PDF. Resumen del requisito: Se ejecutan tres ponderaciones. CountVectorizer(binary=True) marca presencia; binary=False cuenta ocurrencias. TfidfVectorizer pondera por TF-IDF. (1,1) usa unigramas; (1,2) uni+bi; (1,3) uni+bi+tri. El PDF no obliga a probar trigramas solos ni las siete combinaciones del pipeline original. min_df=2, max_df=.95 y max_features=30000 son fijos para comparar; se aprende vocabulario solo en train durante selección.

### Para entenderlo desde cero

El modelo no entiende directamente una oración como una persona. El vectorizador transforma cada noticia en una fila numérica usando un vocabulario de términos. Binaria solo registra si un término aparece; frecuencia registra cuántas veces aparece; TF-IDF combina frecuencia con una ponderación que reduce la importancia relativa de términos extendidos por muchos documentos.

Un n-grama es una secuencia de n tokens consecutivos. (1,1) usa unigramas; (1,2) incluye unigramas y bigramas; (1,3) incluye unigramas, bigramas y trigramas. El primer número no significa el número de clusters. Los n-gramas se generan dentro del vectorizador, antes de construir la matriz.

min_df=2 exige presencia en al menos dos documentos de ajuste. max_df=.95 excluye términos presentes en más del 95% de esos documentos. max_features=30000 limita el vocabulario. Son controles de tamaño y ruido, no valores ganadores seleccionados por separado. La cantidad final de columnas se consulta en vocabulario.tsv y en las matrices, no se supone que siempre se alcance el límite de 30000.

La palabra binaria describe la matriz léxica de presencia/ausencia. Después de SVD y L2 los números pueden dejar de ser 0 o 1; eso es esperado. Las matrices X_words_*.npz conservan la representación léxica anterior a esas transformaciones, mientras K-Means usa el espacio transformado.

Ejemplo: Con "cancer cancer treatment", frecuencia produce cancer=2 y treatment=1; binaria produce 1 y 1. Con "climate change today", (1,2) incluye climate, change, today, climate change y change today.

### Dónde está y qué archivo demuestra el resultado

- resultados/experiments.tsv: Cada fila identifica configuración y resultado dev. Incluye tamaños, elegibilidad, convergencia y selected.

- resultados/vocabulario.tsv: Un término por fila en el mismo orden que las columnas de las matrices NPZ.

- resultados/parametros_pipeline_final.json: Incluye los parámetros de cada etapa del modelo cargado, también opciones por defecto del vectorizador.

### Cómo leer la implementación

- opts reúne los parámetros comunes a las tres representaciones.

- TfidfVectorizer se usa si representation es tfidf.

- CountVectorizer(binary=True) crea presencia/ausencia; False crea conteos.

- ngram_range recibe el rango elegido de GRAMS.

- El vocabulario se aprende al ejecutar fit_transform sobre train, no al construir el objeto.

Archivo y enlace exacto: [experimentos.py](/Users/diamandazurita/Downloads/NLP_News_Project/fase_2_aprendizaje/no_supervisada/experimentos.py:61). Función o contexto: `feature_pipeline`. Líneas 61–68.

```python
def feature_pipeline(cfg):
    opts = dict(ngram_range=tuple(cfg['ngram']), min_df=2, max_df=.95, max_features=30000)
    vect = TfidfVectorizer(**opts) if cfg['representation'] == 'tfidf' else CountVectorizer(binary=cfg['representation'] == 'binaria', **opts)
    steps = [('vectorizador', vect)]
    if cfg['svd']:
        steps.append(('svd', TruncatedSVD(n_components=cfg['svd'], random_state=SEED)))
    steps.append(('norma_l2', Normalizer()))
    return Pipeline(steps)
```

Lectura de las líneas de este fragmento:

- Línea 61: Define la función que se invoca desde otro punto del programa; definirla todavía no ejecuta su contenido.

- Línea 62: Configura los tamaños de secuencias de tokens que se convertirán en características.

- Línea 63: Construye TF-IDF si la representación lo pide; de lo contrario se utiliza el contador de términos.

- Línea 64: Asigna el valor o resultado de la derecha a la variable de la izquierda para usarlo en los pasos siguientes.

- Línea 65: Evalúa la condición y ejecuta esta rama únicamente cuando corresponde.

- Línea 66: Define la reducción de dimensiones y su semilla; los parámetros se aprenden cuando se hace fit.

- Línea 67: Añade normalización numérica L2, diferente de la limpieza lingüística.

- Línea 68: Devuelve el resultado de la función a quien la llamó; no implica que se guarde automáticamente en disco.

Archivo y enlace exacto: [experimentos.py](/Users/diamandazurita/Downloads/NLP_News_Project/fase_2_aprendizaje/no_supervisada/experimentos.py:36). Función o contexto: `nivel principal del módulo`. Líneas 36–37.

```python
REPS = ['binaria', 'frecuencia', 'tfidf']
GRAMS = [(1, 1), (1, 2), (1, 3)]
```

Lectura de las líneas de este fragmento:

- Línea 36: Asigna el valor o resultado de la derecha a la variable de la izquierda para usarlo en los pasos siguientes.

- Línea 37: Asigna el valor o resultado de la derecha a la variable de la izquierda para usarlo en los pasos siguientes.

Para explicarlo con tus palabras: Comparamos tres maneras de poner pesos a los términos y tres rangos de n-gramas para comprobar cómo cambia la similitud textual.

## 6. SVD y normalización de longitud L2

Qué pide el profesor: consultar sección(es) 5 y 7 del PDF. Resumen del requisito: Se comparan sin SVD y 50, 100, 150 dimensiones. L2 escala cada vector a longitud unitaria; no es la normalización lingüística del texto. K-Means y silhouette operan en el mismo espacio euclidiano: léxico+L2 o SVD+L2.

### Para entenderlo desde cero

La matriz original puede tener miles de columnas. TruncatedSVD la resume en un número menor de dimensiones numéricas, conservando parte de su estructura. SVD=50 significa cincuenta dimensiones, no cincuenta temas ni cincuenta clusters. El número de grupos lo controla k.

Después aplicamos Normalizer(), que realiza normalización L2: ajusta la longitud de cada vector a uno, excepto un vector cero, que sigue siendo cero. Así la longitud del texto no domina del mismo modo que en conteos sin normalizar. No es la misma normalización que quitar stopwords: aquella transforma palabras y esta transforma vectores.

Se comparó el mismo tipo de configuración con y sin reducción y con 50/100/150 dimensiones. Cuando hay SVD, los centros viven en ese espacio reducido; para mostrar palabras hay que volver al vocabulario original mediante inverse_transform. Silhouette se calcula sobre el espacio que realmente utilizó K-Means.

Ejemplo: Puedes imaginar una noticia descrita inicialmente por miles de coordenadas de palabras. SVD la representa con 50 coordenadas resumidas. K-Means todavía busca 12 grupos dentro de ese espacio.

### Dónde está y qué archivo demuestra el resultado

- resultados/experiments.tsv: Cada fila identifica configuración y resultado dev. Incluye tamaños, elegibilidad, convergencia y selected.

- resultados/metricas_finales.tsv: Tres filas: train, dev y test. Incluye silhouette, espacio, distancia, tamaños, ARI/NMI externos y vectores cero.

- modelos/kmeans_pipeline.joblib: Objeto con normalizador, vectorizador, SVD=50, L2 y K-Means k=12 ya entrenados con 3948 textos.

### Cómo leer la implementación

- if cfg["svd"] decide si se añade reducción.

- TruncatedSVD(n_components=...) fija cuántas dimensiones conservar.

- random_state=SEED hace reproducible la parte aleatoria de SVD.

- Normalizer se aplica después de SVD o directamente al espacio léxico si no hay SVD.

- Pipeline ejecuta las etapas en ese orden.

Archivo y enlace exacto: [experimentos.py](/Users/diamandazurita/Downloads/NLP_News_Project/fase_2_aprendizaje/no_supervisada/experimentos.py:65). Función o contexto: `feature_pipeline`. Líneas 65–68.

```python
    if cfg['svd']:
        steps.append(('svd', TruncatedSVD(n_components=cfg['svd'], random_state=SEED)))
    steps.append(('norma_l2', Normalizer()))
    return Pipeline(steps)
```

Lectura de las líneas de este fragmento:

- Línea 65: Evalúa la condición y ejecuta esta rama únicamente cuando corresponde.

- Línea 66: Define la reducción de dimensiones y su semilla; los parámetros se aprenden cuando se hace fit.

- Línea 67: Añade normalización numérica L2, diferente de la limpieza lingüística.

- Línea 68: Devuelve el resultado de la función a quien la llamó; no implica que se guarde automáticamente en disco.

Archivo y enlace exacto: [experimentos.py](/Users/diamandazurita/Downloads/NLP_News_Project/fase_2_aprendizaje/no_supervisada/experimentos.py:160). Función o contexto: `run`. Líneas 160–161.

```python
    for svd in [None,50,150]:
        trial({**base,'svd':svd},'svd')
```

Lectura de las líneas de este fragmento:

- Línea 160: Repite el bloque para cada valor de la colección indicada, por ejemplo una configuración, semilla o grupo.

- Línea 161: Continúa la expresión de la línea anterior; hay que leerla dentro de la llamada o estructura completa mostrada arriba.

Para explicarlo con tus palabras: SVD reduce dimensiones; L2 ajusta la longitud del vector; k fija cuántos clusters buscamos. Son decisiones distintas.

## 7. Usar únicamente K-Means y variar hiperparámetros

Qué pide el profesor: consultar sección(es) 7 del PDF. Resumen del requisito: El único algoritmo de agrupamiento es KMeans. Se prueban k=4/6/8/10/12, n_init=1/10/20, init aleatoria y k-means++, max_iter=100/300/500. La búsqueda principal fija random_state=0. No se impone k=10 por haber diez categorías. Se elige una sola rama: LDA no corresponde a esta entrega.

### Para entenderlo desde cero

K-Means busca k centros para agrupar vectores cercanos. Alterna entre asignar cada noticia al centro más cercano y recalcular los centros con las noticias asignadas. El centro es un vector promedio, no necesariamente una noticia que exista en el corpus.

n_clusters controla k. init decide cómo escoger centros iniciales: k-means++ busca una inicialización dispersa y random utiliza una alternativa aleatoria. n_init indica cuántos reinicios se prueban dentro de un ajuste. max_iter pone el límite de iteraciones de cada reinicio; no obliga a agotar ese límite si ya se alcanzó la convergencia.

Dentro de un ajuste, K-Means conserva el reinicio con menor suma de distancias cuadradas a los centros, conocida como inercia. Esa decisión interna es distinta de nuestro criterio externo de selección, silhouette en dev. Por eso aumentar n_init no garantiza un silhouette mayor. Las representaciones Count/TF-IDF, SVD y K-Means son técnicas tradicionales: la tarea de aprendizaje no usa LLMs ni modelos generativos.

La búsqueda es escalonada: primero cruza tres normalizaciones, tres representaciones, tres rangos y tres k, dando 81 candidatos. Después agrega diez ajustes alrededor de buenos candidatos: cambia SVD, prueba k=6/10 y varía parámetros individualmente. No se probaron todas las combinaciones posibles de todos los parámetros, y no se necesita afirmar eso para justificar el experimento.

Ejemplo: La configuración elegida usa k=12, n_init=10 y max_iter=300. El candidato ganador de selección convergió en 15 iteraciones: poner 300 no significa que se ejecutaron 300.

### Dónde está y qué archivo demuestra el resultado

- resultados/experiments.tsv: Cada fila identifica configuración y resultado dev. Incluye tamaños, elegibilidad, convergencia y selected.

- resultados/plan_experimental.json: Registra variantes, controles del vectorizador, etapas, semilla, criterio de admisibilidad y huella del corpus etiquetado.

### Cómo leer la implementación

- kmeans() construye el estimador a partir de cfg.

- itertools.product enumera las 81 combinaciones iniciales.

- trial ejecuta un candidato y lo registra.

- Las siguientes llamadas cambian dimensiones, k o un hiperparámetro.

- SEED=0 se conserva en los candidatos principales; las otras semillas se estudian después como estabilidad.

Archivo y enlace exacto: [experimentos.py](/Users/diamandazurita/Downloads/NLP_News_Project/fase_2_aprendizaje/no_supervisada/experimentos.py:34). Función o contexto: `nivel principal del módulo`. Líneas 34–34.

```python
SEED = 0
```

Lectura de las líneas de este fragmento:

- Línea 34: Asigna el valor o resultado de la derecha a la variable de la izquierda para usarlo en los pasos siguientes.

Archivo y enlace exacto: [experimentos.py](/Users/diamandazurita/Downloads/NLP_News_Project/fase_2_aprendizaje/no_supervisada/experimentos.py:71). Función o contexto: `kmeans`. Líneas 71–72.

```python
def kmeans(cfg, seed=SEED):
    return KMeans(n_clusters=cfg['k'], n_init=cfg['n_init'], init=cfg['init'], max_iter=cfg['max_iter'], random_state=seed)
```

Lectura de las líneas de este fragmento:

- Línea 71: Define la función que se invoca desde otro punto del programa; definirla todavía no ejecuta su contenido.

- Línea 72: Configura cuántos grupos solicita K-Means, junto con los otros hiperparámetros de esta llamada.

Archivo y enlace exacto: [experimentos.py](/Users/diamandazurita/Downloads/NLP_News_Project/fase_2_aprendizaje/no_supervisada/experimentos.py:157). Función o contexto: `run`. Líneas 157–170.

```python
    for mode,rep,ngram,k in itertools.product(NORMS,REPS,GRAMS,[4,8,12]):
        trial(dict(normalization=mode,representation=rep,ngram=ngram,svd=100,k=k,n_init=10,init='k-means++',max_iter=300),'factorial')
    base = cfg_of(best())
    for svd in [None,50,150]:
        trial({**base,'svd':svd},'svd')
    for k in [6,10]:
        trial({**base,'k':k},'k_refinement')
    base = cfg_of(best())
    for value in [1,20]:
        trial({**base,'n_init':value},'n_init')
    trial({**base,'init':'random'},'init')
    for value in [100,500]:
        trial({**base,'max_iter':value},'max_iter')
    winner = best()
```

Lectura de las líneas de este fragmento:

- Línea 157: Repite el bloque para cada valor de la colección indicada, por ejemplo una configuración, semilla o grupo.

- Línea 158: Asigna el valor o resultado de la derecha a la variable de la izquierda para usarlo en los pasos siguientes.

- Línea 159: Toma únicamente la configuración del mejor candidato actual para construir las siguientes comparaciones controladas.

- Línea 160: Repite el bloque para cada valor de la colección indicada, por ejemplo una configuración, semilla o grupo.

- Línea 161: Continúa la expresión de la línea anterior; hay que leerla dentro de la llamada o estructura completa mostrada arriba.

- Línea 162: Repite el bloque para cada valor de la colección indicada, por ejemplo una configuración, semilla o grupo.

- Línea 163: Continúa la expresión de la línea anterior; hay que leerla dentro de la llamada o estructura completa mostrada arriba.

- Línea 164: Toma únicamente la configuración del mejor candidato actual para construir las siguientes comparaciones controladas.

- Línea 165: Repite el bloque para cada valor de la colección indicada, por ejemplo una configuración, semilla o grupo.

- Línea 166: Continúa la expresión de la línea anterior; hay que leerla dentro de la llamada o estructura completa mostrada arriba.

- Línea 167: Continúa la expresión de la línea anterior; hay que leerla dentro de la llamada o estructura completa mostrada arriba.

- Línea 168: Repite el bloque para cada valor de la colección indicada, por ejemplo una configuración, semilla o grupo.

- Línea 169: Continúa la expresión de la línea anterior; hay que leerla dentro de la llamada o estructura completa mostrada arriba.

- Línea 170: Asigna el valor o resultado de la derecha a la variable de la izquierda para usarlo en los pasos siguientes.

Para explicarlo con tus palabras: Utilizamos solo K-Means y variamos sus parámetros. No fijamos k igual a las diez categorías del RSS; k=12 ganó entre los candidatos probados.

## 8. Seleccionar en dev y explicar el criterio

Qué pide el profesor: consultar sección(es) 7 y 9 del PDF. Resumen del requisito: Se exige ocupar los k grupos en dev, al menos 1% de noticias por grupo y máximo 65% en uno solo. Entre candidatos admisibles gana el mayor silhouette sin redondear; el primer ID resuelve empates. Son umbrales de diseño, no garantía de temas puros. La interpretación no se utiliza para ajustar usando test.

### Para entenderlo desde cero

Probar configuraciones no basta: hay que declarar cómo se elige una. Primero filtramos particiones problemáticas en dev: deben aparecer los k grupos, ninguno puede tener menos del 1% de dev y ninguno más del 65%. Estos límites son decisiones del estudio, no umbrales impuestos literalmente por el profesor.

Entre los candidatos que pasan esos controles se elige el mayor silhouette sin redondear. Si empatan, max conserva el primer candidato encontrado. El registro selected identifica únicamente al ganador; los otros candidatos siguen en la tabla para demostrar lo que ocurrió.

La selección no consulta ARI/NMI contra las categorías ni el silhouette de test. Tampoco fuerza el modelo anterior de k=10. La lectura temática posterior sirve para explicar ventajas y limitaciones; no convierte por sí sola al mejor silhouette en el mejor resultado imaginable. Aquí se eligió el mejor bajo el criterio declarado dentro de una búsqueda acotada.

Ejemplo: KM083 ganó con silhouette dev 0.1658002776. Cambiar max_iter a 100 o 500 produjo el mismo valor en esos candidatos; se conservó KM083 por el desempate con el primer ID.

### Dónde está y qué archivo demuestra el resultado

- resultados/seleccion_dev.json: Identifica KM083, sus parámetros, regla de selección y silhouette dev 0.165800.

- resultados/experiments.tsv: Cada fila identifica configuración y resultado dev. Incluye tamaños, elegibilidad, convergencia y selected.

### Cómo leer la implementación

- eligible evalúa ocupación, tamaño mínimo, tamaño máximo y existencia de silhouette.

- best() reúne únicamente las filas elegibles.

- Si no existe candidato elegible, se detiene antes de evaluar test.

- max(... key=silhouette) compara los valores completos, no los redondeados del informe.

- seleccion_dev.json congela la configuración escogida.

Archivo y enlace exacto: [experimentos.py](/Users/diamandazurita/Downloads/NLP_News_Project/fase_2_aprendizaje/no_supervisada/experimentos.py:39). Función o contexto: `nivel principal del módulo`. Líneas 39–40.

```python
MIN_SHARE = .01
MAX_SHARE = .65
```

Lectura de las líneas de este fragmento:

- Línea 39: Asigna el valor o resultado de la derecha a la variable de la izquierda para usarlo en los pasos siguientes.

- Línea 40: Asigna el valor o resultado de la derecha a la variable de la izquierda para usarlo en los pasos siguientes.

Archivo y enlace exacto: [experimentos.py](/Users/diamandazurita/Downloads/NLP_News_Project/fase_2_aprendizaje/no_supervisada/experimentos.py:139). Función o contexto: `run.trial`. Líneas 139–139.

```python
        eligible = q['silhouette'] is not None and q['occupied_clusters'] == cfg['k'] and q['min_share'] >= MIN_SHARE and q['max_share'] <= MAX_SHARE
```

Lectura de las líneas de este fragmento:

- Línea 139: Comprueba que la partición cumpla la regla declarada de ocupación, tamaños y métrica definida.

Archivo y enlace exacto: [experimentos.py](/Users/diamandazurita/Downloads/NLP_News_Project/fase_2_aprendizaje/no_supervisada/experimentos.py:148). Función o contexto: `run.best`. Líneas 148–152.

```python
    def best():
        good = [r for r in records if r['eligible']]
        if not good:
            raise RuntimeError('Ninguna configuración admisible. Revisar resultados dev sin acceder a test.')
        return max(good,key=lambda r:r['silhouette'])
```

Lectura de las líneas de este fragmento:

- Línea 148: Define la función que se invoca desde otro punto del programa; definirla todavía no ejecuta su contenido.

- Línea 149: Filtra los experimentos y conserva solamente los que tienen eligible verdadero.

- Línea 150: Evalúa la condición y ejecuta esta rama únicamente cuando corresponde.

- Línea 151: Detiene el proceso y comunica la condición que impide continuar.

- Línea 152: Elige el mayor silhouette entre los candidatos admisibles, usando sus valores completos.

Archivo y enlace exacto: [experimentos.py](/Users/diamandazurita/Downloads/NLP_News_Project/fase_2_aprendizaje/no_supervisada/experimentos.py:170). Función o contexto: `run`. Líneas 170–174.

```python
    winner = best()
    winner['selected'] = True
    cfg = cfg_of(winner)
    pd.DataFrame(records).to_csv(OUT/'experiments.tsv',sep='\t',index=False)
    write_json(OUT/'seleccion_dev.json',dict(experiment_id=winner['experiment_id'],config=cfg,selection=plan['selection'],silhouette_dev=winner['silhouette']))
```

Lectura de las líneas de este fragmento:

- Línea 170: Asigna el valor o resultado de la derecha a la variable de la izquierda para usarlo en los pasos siguientes.

- Línea 171: Registra o consulta la decisión de selección; no es la categoría original de la noticia.

- Línea 172: Asigna el valor o resultado de la derecha a la variable de la izquierda para usarlo en los pasos siguientes.

- Línea 173: Exporta la tabla al archivo indicado. Esta es la línea que produce la evidencia TSV.

- Línea 174: Guarda en JSON la configuración, decisión o información estructurada que se le entrega.

Para explicarlo con tus palabras: El ganador se decide con dev y un criterio explícito de separación y tamaños; test no participa en esa decisión.

## 9. Silhouette con espacio y distancia explícitos

Qué pide el profesor: consultar sección(es) 7 del PDF. Resumen del requisito: Se usa distancia euclidiana y todos los documentos de dev, sin muestreo. Un resultado más alto describe separación geométrica, no accuracy ni coherencia semántica garantizada. Si solo se ocupa un grupo, la métrica no se calcula y el candidato queda fuera.

### Para entenderlo desde cero

Silhouette compara la cercanía de una noticia con su propio grupo frente a su cercanía con otros grupos. Cerca de 1 suele indicar una asignación bien separada; cerca de 0 sugiere fronteras superpuestas; valores negativos pueden indicar que otros grupos quedan más cerca. No es una tasa de aciertos.

En este proyecto se calcula con distancia euclidiana. Cuando el candidato usa SVD, primero se proyecta y se normaliza L2; sin SVD se utiliza el espacio léxico normalizado. Por eso al informar un silhouette hay que decir en qué espacio y con qué distancia se obtuvo.

La función también cuenta cuántas noticias aparecen en cada grupo y detecta casos donde silhouette no está definido, como tener un solo grupo ocupado. Un valor positivo no certifica que un humano reconozca un tema. El boilerplate de una fuente puede producir un grupo compacto sin coherencia temática.

Ejemplo: Test final dio 0.177708 aproximadamente. No se lee como "17.77% de precisión". Se interpreta como separación positiva pero modesta en el espacio evaluado.

### Dónde está y qué archivo demuestra el resultado

- resultados/experiments.tsv: Cada fila identifica configuración y resultado dev. Incluye tamaños, elegibilidad, convergencia y selected.

- resultados/metricas_finales.tsv: Tres filas: train, dev y test. Incluye silhouette, espacio, distancia, tamaños, ARI/NMI externos y vectores cero.

### Cómo leer la implementación

- np.bincount calcula el tamaño de cada cluster.

- np.count_nonzero indica cuántos grupos quedaron ocupados.

- La condición 1 < n_labels < len(labels) comprueba si silhouette puede calcularse.

- silhouette_score(... metric="euclidean") obtiene la métrica.

- El diccionario resultante guarda tamaños y proporciones para la selección y el análisis.

Archivo y enlace exacto: [experimentos.py](/Users/diamandazurita/Downloads/NLP_News_Project/fase_2_aprendizaje/no_supervisada/experimentos.py:75). Función o contexto: `quality`. Líneas 75–79.

```python
def quality(X, labels, k):
    counts = np.bincount(labels, minlength=k)
    n_labels = np.count_nonzero(counts)
    sil = float(silhouette_score(X, labels, metric='euclidean')) if 1 < n_labels < len(labels) else None
    return dict(silhouette=sil, occupied_clusters=int(n_labels), min_cluster=int(counts.min()), max_cluster=int(counts.max()), min_share=float(counts.min()/len(labels)), max_share=float(counts.max()/len(labels)), sizes=counts.tolist())
```

Lectura de las líneas de este fragmento:

- Línea 75: Define la función que se invoca desde otro punto del programa; definirla todavía no ejecuta su contenido.

- Línea 76: Cuenta noticias por identificador de cluster; minlength incluye también grupos vacíos.

- Línea 77: Cuenta cuántos elementos cumplen la condición numérica, como grupos ocupados o vectores sin conteos.

- Línea 78: Calcula separación geométrica sobre los vectores y grupos indicados; la distancia se fija en euclidean.

- Línea 79: Devuelve el resultado de la función a quien la llamó; no implica que se guarde automáticamente en disco.

Para explicarlo con tus palabras: Usamos silhouette como medida geométrica de separación y la acompañamos con lectura de noticias, porque no equivale a calidad semántica garantizada.

## 10. Comparar estabilidad en dev

Qué pide el profesor: consultar sección(es) 7 del PDF. Resumen del requisito: Se mantienen configuración y SVD, y se ajusta K-Means con semillas 0,1,2. ARI entre pares compara particiones de los mismos documentos, sin categorías gold. No se elige una semilla nueva por su mejor puntuación. Los tres ajustes de estabilidad están en JSON y se incluyen en la bitácora consolidada de 94 ajustes.

### Para entenderlo desde cero

K-Means puede llegar a particiones distintas según los centros iniciales. Estabilidad significa revisar si el patrón se conserva cuando se cambia esa inicialización. Aquí mantenemos las características y los parámetros seleccionados y repetimos K-Means con semillas 0, 1 y 2.

Comparamos las asignaciones de los mismos documentos dev mediante ARI entre pares de ejecuciones. Si los números de cluster se permutan, ARI puede seguir indicando coincidencia: compara la partición, no exige que un grupo conserve su nombre numérico. En esta revisión las categorías reales no intervienen.

Este análisis no busca una nueva semilla ganadora. Aunque la semilla 1 tenga un silhouette algo mayor, conservamos la semilla 0 definida para el modelo principal. Los tres ajustes de estabilidad están aparte de los 91 candidatos y se reúnen en la bitácora de 94 ajustes con candidate_for_selection=False.

Ejemplo: Los ARI entre semillas fueron aproximadamente 0.7625, 0.6870 y 0.6760. Hay coincidencia considerable, pero las particiones no son idénticas. Silhouettes parecidos tampoco implican las mismas asignaciones.

### Dónde está y qué archivo demuestra el resultado

- resultados/estabilidad_dev.json: Tres silhouttes y tamaños dev por semilla, y tres ARI entre pares de asignaciones.

- resultados/experiments_completo.tsv: Añade tres ejecuciones de estabilidad a las 91 de selección. candidate_for_selection separa las dos finalidades.

### Cómo leer la implementación

- assignments guarda un vector de clusters por semilla.

- El bucle ajusta K-Means tres veces con el mismo Xtr.

- predict genera las asignaciones sobre el mismo Xdv.

- combinations enumera los pares 0–1, 0–2 y 1–2.

- adjusted_rand_score compara las dos particiones, no las etiquetas RSS.

Archivo y enlace exacto: [experimentos.py](/Users/diamandazurita/Downloads/NLP_News_Project/fase_2_aprendizaje/no_supervisada/experimentos.py:179). Función o contexto: `run`. Líneas 179–186.

```python
    assignments = {}
    stability = []
    for seed in [0,1,2]:
        km = kmeans(cfg,seed).fit(Xtr)
        labels = km.predict(Xdv)
        assignments[seed] = labels
        stability.append(dict(seed=seed,**quality(Xdv,labels,cfg['k'])))
    write_json(OUT/'estabilidad_dev.json',dict(trials=stability,pairwise_ari=[dict(seed_a=a,seed_b=b,ari=adjusted_rand_score(assignments[a],assignments[b])) for a,b in itertools.combinations(assignments,2)]))
```

Lectura de las líneas de este fragmento:

- Línea 179: Asigna el valor o resultado de la derecha a la variable de la izquierda para usarlo en los pasos siguientes.

- Línea 180: Asigna el valor o resultado de la derecha a la variable de la izquierda para usarlo en los pasos siguientes.

- Línea 181: Repite el bloque para cada valor de la colección indicada, por ejemplo una configuración, semilla o grupo.

- Línea 182: Ajusta K-Means con la matriz de entrenamiento, sin pasar categorías como respuesta.

- Línea 183: Asigna los documentos dev a los centros aprendidos con train.

- Línea 184: Asigna el valor o resultado de la derecha a la variable de la izquierda para usarlo en los pasos siguientes.

- Línea 185: Agrega un elemento a la colección que se exportará o presentará después.

- Línea 186: Compara dos particiones. Revisa los argumentos para distinguir estabilidad entre semillas de análisis externo con categorías.

Para explicarlo con tus palabras: Revisamos sensibilidad a la inicialización con tres semillas y ARI entre particiones; mantenemos la semilla principal sin volver a seleccionar usando ese análisis.

## 11. Congelar y reajustar el pipeline completo

Qué pide el profesor: consultar sección(es) 4, 5, 9; flujo solicitado del PDF. Resumen del requisito: Tras decidir en dev, se reajustan normalizador, vectorizador, SVD, L2 y K-Means con 3948 textos train+dev. Test no participa en fit. La sección 6 exige explícitamente este refit para clasificación; en K-Means se hace además conforme al flujo pedido en esta conversación.

### Para entenderlo desde cero

Después de decidir la configuración, podemos aprovechar los documentos de train y dev juntos para ajustar el modelo que se entregará. Esto se llama reajuste final o refit. Ya no estamos comparando candidatos: los parámetros están fijados y test sigue reservado.

Se construye un Pipeline completo que incluye normalización del texto, vectorizador, SVD si corresponde, L2 y K-Means. final.fit(train_dev.text) reajusta los componentes que aprenden. En particular no se reutiliza sin cambios un vocabulario que solo vio train: se aprende el vocabulario final con las 3948 noticias permitidas.

El modelo se serializa con joblib para usarlo después sin entrenar otra vez. Reajustar puede cambiar los centros, el vocabulario, los tamaños y los números de los grupos. Por eso las métricas dev del modelo final no son las métricas independientes con las que se eligió KM083: esas noticias ya entraron al ajuste final.

Ejemplo: En selección, dev obtuvo 0.165800. En el modelo final reajustado, dev dio 0.154702. No es una contradicción: son objetos ajustados con conjuntos distintos, y el segundo dev ya no es independiente.

### Dónde está y qué archivo demuestra el resultado

- modelos/kmeans_pipeline.joblib: Objeto con normalizador, vectorizador, SVD=50, L2 y K-Means k=12 ya entrenados con 3948 textos.

- resultados/configuracion_final.json: Guarda configuración ganadora, 3948 textos de ajuste, versiones, semilla y test_used_for_selection=False.

- resultados/parametros_pipeline_final.json: Incluye los parámetros de cada etapa del modelo cargado, también opciones por defecto del vectorizador.

### Cómo leer la implementación

- isin(["train","dev"]) reúne las dos particiones permitidas.

- Pipeline incorpora el normalizador y las etapas numéricas.

- final.fit aprende con train_dev.text, sin textos test.

- joblib.dump guarda el objeto entrenado.

- configuracion_final.json documenta qué se eligió y qué particiones se usaron para ajustar.

Archivo y enlace exacto: [experimentos.py](/Users/diamandazurita/Downloads/NLP_News_Project/fase_2_aprendizaje/no_supervisada/experimentos.py:188). Función o contexto: `run`. Líneas 188–193.

```python
    train_dev = docs[docs.split.isin(['train','dev'])].reset_index(drop=True)
    final = Pipeline([('normalizacion',NormalizadorNoticias(modo=cfg['normalization'])),*feature_pipeline(cfg).steps,('kmeans',kmeans(cfg))])
    print('Ajuste final del pipeline completo en train + dev:',cfg,flush=True)
    final.fit(train_dev.text)
    joblib.dump(final,MOD/'kmeans_pipeline.joblib')
    write_json(OUT/'configuracion_final.json',dict(selected_experiment=winner['experiment_id'],config=cfg,train_dev_count=len(train_dev),test_used_for_selection=False,vectorizer_refitted_on='train+dev',source_sha256=plan['source_sha256'],python=platform.python_version(),versions={p:version(p) for p in ['numpy','pandas','scipy','scikit-learn','spacy','en-core-web-sm','joblib','reportlab','matplotlib','threadpoolctl']},seed=SEED))
```

Lectura de las líneas de este fragmento:

- Línea 188: Filtra, conserva o comprueba la partición de los documentos; no la convierte en característica del texto.

- Línea 189: Asigna el valor o resultado de la derecha a la variable de la izquierda para usarlo en los pasos siguientes.

- Línea 190: Muestra información al ejecutar la consola; no cambia el ajuste del modelo.

- Línea 191: Reajusta el pipeline completo con los textos de train y dev juntos; test queda fuera.

- Línea 192: Guarda el objeto ajustado en disco para poder cargarlo después sin volver a entrenar.

- Línea 193: Guarda en JSON la configuración, decisión o información estructurada que se le entrega.

Para explicarlo con tus palabras: Primero seleccionamos con dev; después reajustamos el pipeline completo con train+dev y guardamos ese modelo para la evaluación y el uso final.

## 12. Asignar documentos reservados con predict

Qué pide el profesor: consultar sección(es) 7 y 11 del PDF. Resumen del requisito: Se aplican únicamente transformaciones del modelo ya ajustado y predict. Se conservan IDs y split al guardar los clusters. La auditoría recalcula la misma evaluación guardada para comprobar consistencia: no vuelve a entrenar K-Means ni elige parámetros con test.

### Para entenderlo desde cero

Predecir aquí significa asignar una noticia a uno de los centros aprendidos, no predecir una categoría supervisada. Durante esta fase no se crean nuevos centros a partir de test. Se transforman sus textos usando el vocabulario y SVD ya ajustados y se elige el centro más cercano.

El script transforma todo el corpus con el modelo final y conserva el orden de los documentos para exportar los resultados. La máscara split permite separar train, dev y test al calcular sus métricas. El archivo de asignaciones contiene ID, partición, cluster y configuración, lo que permite rastrear una noticia sin guardar las categorías como parte de la entrada del modelo.

La auditoría comprueba de nuevo la misma evaluación almacenada: verifica que el modelo cargado reproduce los clusters y que el silhouette coincide. Eso no añade una nueva configuración ni justifica retocar el modelo al mirar test. Es una comprobación de consistencia, no una nueva búsqueda.

Ejemplo: En resultados/clustering_results.tsv puedes buscar document_id=N2256 y ver cluster_id=2. La columna split te dice a qué partición pertenece y method/configuration identifica el modelo que hizo la asignación.

### Dónde está y qué archivo demuestra el resultado

- resultados/clustering_results.tsv: 4645 filas con document_id, split, cluster_id y method/configuration.

- resultados/metricas_finales.tsv: Tres filas: train, dev y test. Incluye silhouette, espacio, distancia, tamaños, ARI/NMI externos y vectores cero.

### Cómo leer la implementación

- final["normalizacion"].transform prepara el texto con la configuración ganadora.

- El vectorizador aplica el vocabulario aprendido, sin fit.

- SVD y L2 también usan sus objetos ya ajustados.

- final["kmeans"].predict asigna un cluster por fila.

- results conserva ID y split antes de añadir cluster_id y guardar el TSV.

Archivo y enlace exacto: [experimentos.py](/Users/diamandazurita/Downloads/NLP_News_Project/fase_2_aprendizaje/no_supervisada/experimentos.py:195). Función o contexto: `run`. Líneas 195–205.

```python
    normalized_all = final['normalizacion'].transform(docs.text)
    Xwords = final['vectorizador'].transform(normalized_all)
    X = Xwords
    if 'svd' in final.named_steps:
        X = final['svd'].transform(X)
    X = final['norma_l2'].transform(X)
    labels = final['kmeans'].predict(X)
    results = docs[['document_id','split']].copy()
    results['cluster_id'] = labels
    results['method/configuration'] = json.dumps(dict(method='KMeans',**cfg,random_state=SEED),ensure_ascii=False)
    results.to_csv(OUT/'clustering_results.tsv',sep='\t',index=False)
```

Lectura de las líneas de este fragmento:

- Línea 195: Asigna el valor o resultado de la derecha a la variable de la izquierda para usarlo en los pasos siguientes.

- Línea 196: Asigna el valor o resultado de la derecha a la variable de la izquierda para usarlo en los pasos siguientes.

- Línea 197: Asigna el valor o resultado de la derecha a la variable de la izquierda para usarlo en los pasos siguientes.

- Línea 198: Evalúa la condición y ejecuta esta rama únicamente cuando corresponde.

- Línea 199: Asigna el valor o resultado de la derecha a la variable de la izquierda para usarlo en los pasos siguientes.

- Línea 200: Asigna el valor o resultado de la derecha a la variable de la izquierda para usarlo en los pasos siguientes.

- Línea 201: Asigna el valor o resultado de la derecha a la variable de la izquierda para usarlo en los pasos siguientes.

- Línea 202: Filtra, conserva o comprueba la partición de los documentos; no la convierte en característica del texto.

- Línea 203: Asigna el valor o resultado de la derecha a la variable de la izquierda para usarlo en los pasos siguientes.

- Línea 204: Fija la semilla usada por la etapa para reproducir la parte aleatoria.

- Línea 205: Exporta la tabla al archivo indicado. Esta es la línea que produce la evidencia TSV.

Para explicarlo con tus palabras: Asignamos test mediante predict del modelo ya ajustado; no usamos test para mover centros ni para decidir la configuración.

## 13. Reportar tamaño de cada grupo por partición

Qué pide el profesor: consultar sección(es) 7 y 11 del PDF. Resumen del requisito: La tabla separa train, dev y test y añade total. Suma 4645 noticias. Así no se confunde el tamaño solo de train con el del corpus completo.

### Para entenderlo desde cero

El tamaño de un grupo es el número de noticias que contiene. No basta con decir que existen doce grupos: debemos comprobar si uno absorbe casi todo, si otros quedan vacíos o si aparecen agrupamientos demasiado pequeños para interpretarlos.

cluster_sizes.tsv separa tamaños de train, dev y test y añade total. Los tamaños finales no son los mismos que los tamaños dev de un candidato durante la selección, porque el modelo final se reajustó con train+dev. La suma de todos los totales debe ser 4645 y la suma por columna debe coincidir con cada partición.

La gráfica apila train, dev y test para comparar los grupos de un vistazo. Un grupo grande no es automáticamente incorrecto: puede reflejar una cobertura frecuente; sin embargo, hay que leerlo para saber si es un tema o una bolsa de noticias genéricas. El mayor grupo final se describe honestamente como mixto.

Ejemplo: El cluster 0 reúne 680 train, 131 dev y 140 test: total 951. El menor total final es el cluster 8 con 183. Ninguno quedó vacío.

### Dónde está y qué archivo demuestra el resultado

- resultados/cluster_sizes.tsv: Doce filas, una por grupo, con train/dev/test y total.

- resultados/figuras/tamanos_clusters.png: Barras apiladas de train/dev/test para los doce grupos.

### Cómo leer la implementación

- pd.crosstab cruza cluster_id con split y cuenta filas.

- reindex garantiza que aparezcan todos los IDs de grupo esperados.

- sum(axis=1) suma train, dev y test para cada grupo.

- to_csv guarda la tabla que lee el informe.

- La auditoría compara las sumas con las asignaciones para detectar pérdidas o duplicaciones.

Archivo y enlace exacto: [experimentos.py](/Users/diamandazurita/Downloads/NLP_News_Project/fase_2_aprendizaje/no_supervisada/experimentos.py:213). Función o contexto: `run`. Líneas 213–215.

```python
    counts = pd.crosstab(results.cluster_id,results.split).reindex(range(cfg['k']),fill_value=0)
    counts['total'] = counts.sum(axis=1)
    counts.to_csv(OUT/'cluster_sizes.tsv',sep='\t')
```

Lectura de las líneas de este fragmento:

- Línea 213: Cruza dos variables y cuenta cuántos documentos hay en cada combinación.

- Línea 214: Asigna el valor o resultado de la derecha a la variable de la izquierda para usarlo en los pasos siguientes.

- Línea 215: Exporta la tabla al archivo indicado. Esta es la línea que produce la evidencia TSV.

Para explicarlo con tus palabras: Mostramos tamaños por partición y total; así revisamos balance y evitamos presentar solo train como si fuera todo el corpus.

## 14. Diez palabras representativas en el espacio original

Qué pide el profesor: consultar sección(es) 7 y 11 del PDF. Resumen del requisito: Con SVD, inverse_transform reconstruye el centroide en las columnas del vocabulario original. Se ordenan los pesos y se toman diez términos; con n-gramas un término puede tener varias palabras. No son probabilidades de LDA ni nombres de componentes SVD. Hay 120 filas: diez por doce grupos.

### Para entenderlo desde cero

Las palabras representativas ayudan a entender qué coordenadas pesan más en un centroide. Si usamos SVD, los centros están en dimensiones reducidas que no tienen nombres de palabras. Por eso primero reconstruimos sus pesos en las columnas del vocabulario original.

inverse_transform realiza esa reconstrucción aproximada. Después get_feature_names_out obtiene los términos reales de las columnas y argsort ordena sus pesos. Se toman los diez de mayor valor para cada cluster. Es importante no mostrar el nombre de una componente latente como si fuera una palabra.

Como el vocabulario usa unigramas y bigramas, puede aparecer jensen y también jensen huang. Son diez términos o características, no necesariamente diez conceptos independientes. Tampoco son probabilidades de un modelo LDA. Su lectura debe complementarse con los documentos, porque un término frecuente puede venir del formato de la fuente.

Ejemplo: El cluster 1 incluye ai, nvidia, company, huang, jensen y jensen huang, lo que sugiere IA y negocios tecnológicos. Otro grupo incluye doi y published online: eso puede indicar formato bibliográfico, no un tema científico único.

### Dónde está y qué archivo demuestra el resultado

- resultados/top_words.tsv: 120 filas: doce grupos por diez posiciones. Los términos se obtuvieron en el espacio original de palabras.

- resultados/vocabulario.tsv: Un término por fila en el mismo orden que las columnas de las matrices NPZ.

### Cómo leer la implementación

- svd.inverse_transform devuelve los centroides al espacio de palabras.

- get_feature_names_out da el nombre de cada columna.

- argsort()[::-1] ordena de mayor a menor peso.

- [:10] selecciona diez índices.

- top_rows asocia cada término con cluster_id y rank antes de exportar.

Archivo y enlace exacto: [experimentos.py](/Users/diamandazurita/Downloads/NLP_News_Project/fase_2_aprendizaje/no_supervisada/experimentos.py:82). Función o contexto: `lexical_summary`. Líneas 82–85.

```python
def lexical_summary(vect, svd, centers):
    centers = svd.inverse_transform(centers) if svd is not None else centers
    vocab = vect.get_feature_names_out()
    return [[str(vocab[i]) for i in center.argsort()[::-1][:10]] for center in centers]
```

Lectura de las líneas de este fragmento:

- Línea 82: Define la función que se invoca desde otro punto del programa; definirla todavía no ejecuta su contenido.

- Línea 83: Reconstruye el centroide desde SVD hacia las columnas del vocabulario original.

- Línea 84: Recupera los nombres de los términos en el mismo orden que las columnas de la matriz.

- Línea 85: Ordena los pesos del centro de mayor a menor, conserva diez posiciones y recupera sus términos del vocabulario.

Archivo y enlace exacto: [experimentos.py](/Users/diamandazurita/Downloads/NLP_News_Project/fase_2_aprendizaje/no_supervisada/experimentos.py:220). Función o contexto: `run`. Líneas 220–221.

```python
        for rank, word in enumerate(words[c],1):
            top_rows.append(dict(cluster_id=c,rank=rank,term=word,space='centroid inverse-projected to original word features' if cfg['svd'] else 'word-feature centroid'))
```

Lectura de las líneas de este fragmento:

- Línea 220: Repite el bloque para cada valor de la colección indicada, por ejemplo una configuración, semilla o grupo.

- Línea 221: Añade un término representativo con su cluster y posición dentro del top diez.

Para explicarlo con tus palabras: Extraemos diez términos por centroide en el espacio léxico original, reconstruyéndolo desde SVD cuando corresponde, y los interpretamos con noticias reales.

## 15. Dos noticias representativas y un caso de frontera

Qué pide el profesor: consultar sección(es) 7 y 11 del PDF. Resumen del requisito: Se toman las dos noticias train+dev más cercanas a su propio centroide; se exportan texto completo, URL, distancia, ID y split. Una tercera noticia, la más lejana del grupo, sirve para analizar fronteras. No se usa test para escoger estas noticias de interpretación.

### Para entenderlo desde cero

Una noticia representativa se elige por una regla reproducible: debe pertenecer al grupo y estar cerca de su centro. No se escoge a mano una noticia bonita para hacer parecer coherente un cluster mezclado. Aquí se seleccionan las dos distancias menores dentro de cada grupo entre las noticias de train+dev.

También se exporta una noticia de frontera: la más lejana al centro entre los documentos de ese grupo. Sirve para discutir ruido, mezcla o límites del tema. Frontera no quiere decir error supervisado; no tenemos una respuesta de categoría que el modelo debiera acertar.

El archivo guarda el texto completo y la URL, además del ID, distancia y split. El PDF puede mostrar extractos para no hacerse enorme. Una distancia menor solo tiene sentido comparada dentro del mismo espacio y modelo; no indica mayor importancia periodística ni que esa noticia sea la mejor explicación humana del tema.

Ejemplo: Para el cluster 1 se exportaron N0648 (caída de acciones tecnológicas) y N2645 (filial de NPCI para IA/blockchain). N2517, sobre una salida a bolsa, aparece como caso de frontera del grupo.

### Dónde está y qué archivo demuestra el resultado

- resultados/representative_documents.tsv: 36 filas: dos representantes y una frontera por grupo. Guarda texto, URL, distancia, ID y partición.

### Cómo leer la implementación

- ids filtra documentos asignados al cluster y excluye test.

- distances se calcula entre vectores y centros.

- argsort(... )[:2] identifica las dos noticias más cercanas.

- [-1:] conserva la más lejana como frontera.

- rep_rows añade texto, URL e ID para poder comprobar qué se está mostrando.

Archivo y enlace exacto: [experimentos.py](/Users/diamandazurita/Downloads/NLP_News_Project/fase_2_aprendizaje/no_supervisada/experimentos.py:223). Función o contexto: `run`. Líneas 223–228.

```python
        ids = np.flatnonzero((labels==c) & docs.split.ne('test').to_numpy())
        near = ids[np.argsort(distances[ids,c])[:2]]
        far = ids[np.argsort(distances[ids,c])[-1:]]
        for role, indices in [('representativo',near),('frontera',far)]:
            for rank,i in enumerate(indices,1):
                rep_rows.append(dict(cluster_id=c,role=role,rank=rank,document_id=docs.iloc[i].document_id,split=docs.iloc[i].split,distance=float(distances[i,c]),text=docs.iloc[i].text,url=docs.iloc[i].url))
```

Lectura de las líneas de este fragmento:

- Línea 223: Obtiene posiciones de filas que cumplen la máscara; no son los IDs textuales N0001, N0002, etcétera.

- Línea 224: Ordena de menor a mayor distancia al centroide y toma las primeras dos posiciones: las noticias más cercanas de ese grupo.

- Línea 225: Toma la última posición del orden creciente de distancias: la noticia más lejana del grupo, usada como frontera.

- Línea 226: Procesa primero las posiciones de las dos noticias representativas y después la posición del caso de frontera.

- Línea 227: Recorre las posiciones seleccionadas y les asigna una numeración desde 1 para guardarlas ordenadas.

- Línea 228: Añade ID, texto, URL, distancia y partición de la noticia al listado de representantes o frontera.

Archivo y enlace exacto: [experimentos.py](/Users/diamandazurita/Downloads/NLP_News_Project/fase_2_aprendizaje/no_supervisada/experimentos.py:230). Función o contexto: `run`. Líneas 230–230.

```python
    pd.DataFrame(rep_rows).to_csv(OUT/'representative_documents.tsv',sep='\t',index=False)
```

Lectura de las líneas de este fragmento:

- Línea 230: Exporta la tabla al archivo indicado. Esta es la línea que produce la evidencia TSV.

Para explicarlo con tus palabras: Damos dos documentos más cercanos al centro por grupo y un caso lejano para explicar qué representa el cluster y dónde se vuelve menos coherente.

## 16. Interpretación, temas, grupos mezclados/pequeños/grandes

Qué pide el profesor: consultar sección(es) 7 y 12 del PDF. Resumen del requisito: La interpretación textual está separada de las asignaciones numéricas: cada grupo tiene un título propuesto y un análisis basado en sus términos y noticias. También se examinan experimentos rechazados: KM031 tiene un grupo de 467/697 dev; KM082 uno de cuatro. Los grupos finales por formato o vocabulario temporal se reconocen como limitaciones, no como temas puros.

### Para entenderlo desde cero

El algoritmo devuelve grupos numéricos, no títulos temáticos. La interpretación es la lectura posterior de palabras, noticias cercanas, casos de frontera y tamaños. El archivo interpretacion_manual.json conserva esa lectura del equipo y el generador la incorpora al informe sin alterar asignaciones.

Hay grupos temáticos relativamente claros, como tecnología/IA, política estadounidense, deportes o hallazgos científicos. Otros se explican mejor por términos genéricos, fechas o estilo editorial. Reconocer eso no invalida haber ejecutado K-Means: es parte del análisis que pide el profesor.

Se inspeccionan también configuraciones rechazadas para comprender por qué no se eligieron. KM031 tuvo un grupo de 467 documentos dev y KM082 uno de cuatro. El modelo final no tiene grupos diminutos, pero el requisito de revisar problemas pequeños y grandes se cubre con estos ejemplos reales y la distribución final.

Ejemplo: El cluster 3 final mezcla un estudio sobre un reloj para detectar hipertensión y otro sobre neutralidad de carbono en China. Comparten Nature y DOI, no el mismo tema. Por eso se describe como sesgo de fuente, no como "medicina".

### Dónde está y qué archivo demuestra el resultado

- interpretacion_manual.json: Propone nombre y análisis de cada cluster, comenta estabilidad y documenta pares de eventos. No contiene respuestas objetivo para entrenar.

- resultados/INFORME_NO_SUPERVISADO.md: Metodología, resultados, análisis y limitaciones, con autoría conjunta.

- resultados/INFORME_NO_SUPERVISADO.pdf: Versión de presentación con gráficas y evidencia real.

### Cómo leer la implementación

- notes carga la interpretación escrita del equipo.

- El bucle recorre cada cluster que aparece en la tabla de tamaños.

- Se añade el título propuesto, el análisis y los diez términos.

- Las noticias se extraen del TSV de representantes, no se inventan.

- El generador presenta todo en Markdown y PDF para estudiar y exponer.

Archivo y enlace exacto: [generar_informe.py](/Users/diamandazurita/Downloads/NLP_News_Project/fase_2_aprendizaje/no_supervisada/generar_informe.py:119). Función o contexto: `generate`. Líneas 119–127.

```python
    for c in sizes.cluster_id:
        size=sizes[sizes.cluster_id.eq(c)].iloc[0]
        note=notes['clusters'][str(c)]
        add(f"### Cluster {c}: {note['title']}")
        add(f'Tamaño: train={size["train"]}, dev={size["dev"]}, test={size["test"]}, total={size.total}.')
        add('Diez términos: '+', '.join(words[words.cluster_id.eq(c)].term)+'.')
        add(note['analysis'])
        for r in reps[reps.cluster_id.eq(c)].itertuples():
            add(f'- {r.role} {r.rank}: {r.document_id} ({r.split}), distancia={fmt(r.distance)}. Extracto: {r.text[:900]} URL: {r.url}')
```

Lectura de las líneas de este fragmento:

- Línea 119: Repite el bloque para cada valor de la colección indicada, por ejemplo una configuración, semilla o grupo.

- Línea 120: Asigna el valor o resultado de la derecha a la variable de la izquierda para usarlo en los pasos siguientes.

- Línea 121: Asigna el valor o resultado de la derecha a la variable de la izquierda para usarlo en los pasos siguientes.

- Línea 122: Añade este texto o resultado a la versión escrita del informe.

- Línea 123: Añade este texto o resultado a la versión escrita del informe.

- Línea 124: Añade este texto o resultado a la versión escrita del informe.

- Línea 125: Añade este texto o resultado a la versión escrita del informe.

- Línea 126: Repite el bloque para cada valor de la colección indicada, por ejemplo una configuración, semilla o grupo.

- Línea 127: Filtra, conserva o comprueba la partición de los documentos; no la convierte en característica del texto.

Para explicarlo con tus palabras: Los nombres de temas son interpretaciones justificadas con documentos; también explicamos grupos genéricos y sesgos de fuente en lugar de ocultarlos.

## 17. Comprobar agrupamiento de eventos recurrentes

Qué pide el profesor: consultar sección(es) 7 del PDF. Resumen del requisito: Se comparan pares de IDs sobre el mismo evento y se verifica si su cluster coincide. Ejemplo favorable: N2256 y N2295, sobre los artículos contra Hegseth, quedan en el grupo 2. El video N2250 del mismo suceso queda en el 5: un fallo relacionado con formato. Los ejemplos usan train/dev.

### Para entenderlo desde cero

El objetivo incluye revisar si noticias sobre un mismo evento quedan juntas. Compartir una categoría amplia no basta: dos noticias deportivas pueden tratar partidos distintos. Para examinar eventos elegimos pares concretos de IDs y leemos qué ocurrió en ambos textos.

El generador cruza esos pares con clustering_results.tsv para obtener sus grupos reales. La columna same_cluster permite ver si quedaron juntos. Los ejemplos se toman de train/dev para hacer la lectura cualitativa sin abrir una nueva elección de parámetros a partir de test.

Hay éxitos y fallos. Un mismo evento puede separarse por diferencias de vocabulario, longitud o fórmulas editoriales. Estos ejemplos no son una evaluación exhaustiva de recuperación de eventos ni una métrica de exactitud anotada por humanos: son evidencia cualitativa rastreable que responde a la pregunta del profesor.

Ejemplo: N2256 de NPR y N2295 de CBS describen los artículos de impeachment contra Hegseth y quedan en cluster 2. N2250, la cobertura en video del mismo asunto, queda en cluster 5. El formato puede interferir con el agrupamiento del evento.

### Dónde está y qué archivo demuestra el resultado

- resultados/eventos_recurrentes.tsv: Tres casos con IDs, grupos, same_cluster e interpretación.

- interpretacion_manual.json: Propone nombre y análisis de cada cluster, comenta estabilidad y documenta pares de eventos. No contiene respuestas objetivo para entrenar.

### Cómo leer la implementación

- notes["event_cases"] contiene los pares analizados.

- assigned indexa las asignaciones por document_id.

- loc[a] y loc[b] recuperan el grupo de cada noticia.

- ca == cb informa si el par cayó en el mismo grupo.

- El TSV exporta IDs, grupos y explicación para verificar la conclusión.

Archivo y enlace exacto: [generar_informe.py](/Users/diamandazurita/Downloads/NLP_News_Project/fase_2_aprendizaje/no_supervisada/generar_informe.py:132). Función o contexto: `generate`. Líneas 132–138.

```python
    for case in notes['event_cases']:
        a,b=case['document_a'],case['document_b']
        ca,cb=int(assigned.loc[a,'cluster_id']),int(assigned.loc[b,'cluster_id'])
        assert docs.set_index('document_id').loc[[a,b],'split'].ne('test').all()
        event_rows.append(dict(document_a=a,document_b=b,cluster_a=ca,cluster_b=cb,same_cluster=ca==cb,interpretation=case['interpretation']))
        add(f'- {a} (grupo {ca}) y {b} (grupo {cb}): '+case['interpretation'])
    pd.DataFrame(event_rows).to_csv(OUT/'eventos_recurrentes.tsv',sep='\t',index=False)
```

Lectura de las líneas de este fragmento:

- Línea 132: Repite el bloque para cada valor de la colección indicada, por ejemplo una configuración, semilla o grupo.

- Línea 133: Asigna el valor o resultado de la derecha a la variable de la izquierda para usarlo en los pasos siguientes.

- Línea 134: Asigna el valor o resultado de la derecha a la variable de la izquierda para usarlo en los pasos siguientes.

- Línea 135: Exige que la condición sea verdadera. Si no lo es, la ejecución se detiene con AssertionError.

- Línea 136: Agrega un elemento a la colección que se exportará o presentará después.

- Línea 137: Añade este texto o resultado a la versión escrita del informe.

- Línea 138: Exporta la tabla al archivo indicado. Esta es la línea que produce la evidencia TSV.

Para explicarlo con tus palabras: Comprobamos eventos concretos: mostramos casos que sí se reunieron y coberturas del mismo hecho que se separaron, con sus IDs y grupos reales.

## 18. ARI/NMI y contingencia solo externos

Qué pide el profesor: consultar sección(es) 4 y 7 del PDF. Resumen del requisito: Se comparan los grupos finales con las categorías de origen únicamente para interpretación. Estas métricas no intervienen en best(). ARI externo con category es distinto del ARI entre semillas usado en estabilidad.

### Para entenderlo desde cero

Una vez formado el modelo podemos comparar los clusters con las categorías del RSS como referencia externa. ARI analiza acuerdo entre particiones ajustando por azar; NMI mide información compartida entre ambas. No requieren que los números de cluster sean iguales a nombres o números de categoría.

Estas categorías pueden ser imperfectas o demasiado amplias para los eventos. Un grupo de IA y negocios puede mezclar economía y tecnología, aunque resulte razonable por similitud textual. Por eso un ARI modesto no es automáticamente un error de implementación, pero sí debe explicarse.

La tabla de contingencia cuenta cuántas noticias de cada categoría hay en cada cluster. Sirve para ver grupos mixtos y concentraciones. Es distinta de una matriz de confusión supervisada: los clusters no son predicciones directas de las diez clases y no hay que calcular accuracy haciendo coincidir IDs numéricos arbitrarios.

Ejemplo: En test se obtuvo ARI externo cercano a 0.09985 y NMI cercano a 0.23064. Esos valores describen relación con las categorías; no fueron el criterio para escoger KM083.

### Dónde está y qué archivo demuestra el resultado

- resultados/metricas_finales.tsv: Tres filas: train, dev y test. Incluye silhouette, espacio, distancia, tamaños, ARI/NMI externos y vectores cero.

- resultados/contingencia_train.tsv: Recuento de categorías RSS por cluster para train. No son entradas de aprendizaje ni una matriz de confusión supervisada.

- resultados/contingencia_dev.tsv: Recuento de categorías RSS por cluster para dev. No son entradas de aprendizaje ni una matriz de confusión supervisada.

- resultados/contingencia_test.tsv: Recuento de categorías RSS por cluster para test. No son entradas de aprendizaje ni una matriz de confusión supervisada.

### Cómo leer la implementación

- Las máscaras split separan la evidencia por partición.

- adjusted_rand_score compara category con cluster_id.

- normalized_mutual_info_score realiza la otra comparación externa.

- pd.crosstab cuenta categorías dentro de cada cluster.

- best() no lee estas métricas ni las tablas de contingencia.

Archivo y enlace exacto: [experimentos.py](/Users/diamandazurita/Downloads/NLP_News_Project/fase_2_aprendizaje/no_supervisada/experimentos.py:206). Función o contexto: `run`. Líneas 206–212.

```python
    metrics = []
    for split in ['train','dev','test']:
        mask = docs.split.eq(split).to_numpy()
        q = quality(X[mask],labels[mask],cfg['k'])
        metrics.append(dict(split=split,model='final_refit_train_dev',feature_space=f"svd={cfg['svd']} + L2",distance='euclidean',**q,ari=adjusted_rand_score(docs.loc[mask,'category'],labels[mask]),nmi=normalized_mutual_info_score(docs.loc[mask,'category'],labels[mask]),zero_vectors=int(np.sum(Xwords[mask].getnnz(axis=1)==0))))
        pd.crosstab(pd.Series(labels[mask],name='cluster_id'),pd.Series(docs.loc[mask,'category'].to_numpy(),name='category')).reindex(range(cfg['k']),fill_value=0).to_csv(OUT/f'contingencia_{split}.tsv',sep='\t')
    pd.DataFrame(metrics).to_csv(OUT/'metricas_finales.tsv',sep='\t',index=False)
```

Lectura de las líneas de este fragmento:

- Línea 206: Asigna el valor o resultado de la derecha a la variable de la izquierda para usarlo en los pasos siguientes.

- Línea 207: Filtra, conserva o comprueba la partición de los documentos; no la convierte en característica del texto.

- Línea 208: Filtra, conserva o comprueba la partición de los documentos; no la convierte en característica del texto.

- Línea 209: Asigna el valor o resultado de la derecha a la variable de la izquierda para usarlo en los pasos siguientes.

- Línea 210: Compara dos particiones. Revisa los argumentos para distinguir estabilidad entre semillas de análisis externo con categorías.

- Línea 211: Cruza dos variables y cuenta cuántos documentos hay en cada combinación.

- Línea 212: Exporta la tabla al archivo indicado. Esta es la línea que produce la evidencia TSV.

Para explicarlo con tus palabras: ARI, NMI y contingencia son análisis externo posterior. No entrenamos para imitar las categorías y no elegimos la configuración con esas etiquetas.

## 19. Bitácora completa y comparaciones de un factor

Qué pide el profesor: consultar sección(es) 9 del PDF. Resumen del requisito: experiments.tsv conserva 91 candidatos de selección; experiments_completo.tsv reúne esos 91 más los tres ajustes de estabilidad (94) sin convertir las semillas adicionales en candidatos. comparaciones_controladas.tsv contiene pares del factorial que mantienen todos los otros factores. El informe explica mejoras, empeoramientos y resultados iguales.

### Para entenderlo desde cero

Una bitácora es la tabla de lo que realmente se ejecutó: identifica configuración, resultado y selección. experiments.tsv contiene los 91 candidatos, incluso los que dieron peor resultado o se rechazaron por tamaño. Si solo guardáramos el ganador no podríamos justificar la comparación.

experiments_completo.tsv añade los tres ajustes de estabilidad: total 94 filas. candidate_for_selection diferencia los candidatos de las repeticiones con otras semillas. El reajuste final se documenta por separado en configuracion_final.json y en el modelo; no se confunde con un candidato dev independiente.

Para explicar efectos de una decisión se comparan pares que solo cambian un factor. comparaciones_controladas.tsv contiene esas parejas dentro del factorial. Comparar una configuración con stopwords+TF-IDF+12 grupos frente a otra mínima+conteos+4 grupos no permite atribuir toda diferencia a stopwords. Las medias por factor son descriptivas, no pruebas de causalidad.

Ejemplo: KM033 y KM083 comparten la configuración base pertinente y cambian SVD 100 a 50: el silhouette dev pasa de 0.1117 a 0.1658. En cambio aumentar max_iter a 500 no mejoró el valor del candidato base.

### Dónde está y qué archivo demuestra el resultado

- resultados/experiments.tsv: Cada fila identifica configuración y resultado dev. Incluye tamaños, elegibilidad, convergencia y selected.

- resultados/experiments_completo.tsv: Añade tres ejecuciones de estabilidad a las 91 de selección. candidate_for_selection separa las dos finalidades.

- resultados/comparaciones_controladas.tsv: Dos candidatos con los demás factores constantes y delta de silhouette B-A.

- resultados/resumen_factores.tsv: Medias de silhouette y número de admisibles por valor de factor.

### Cómo leer la implementación

- row registra parámetros, métrica, tamaños y controles para cada ejecución.

- to_csv actualiza la bitácora durante el proceso.

- groupby(others) reúne candidatos iguales en los factores que deben mantenerse.

- Los pares comparan dos valores de un factor dentro de ese grupo.

- consolidate_log añade las semillas como estabilidad, no como nuevas candidatas.

Archivo y enlace exacto: [experimentos.py](/Users/diamandazurita/Downloads/NLP_News_Project/fase_2_aprendizaje/no_supervisada/experimentos.py:141). Función o contexto: `run.trial`. Líneas 141–145.

```python
        row = dict(experiment_id=exp_id,task='clustering',method='KMeans',stage=stage,**cfg,**q,eligible=eligible,selected=False,random_state=SEED,n_iter=int(km.n_iter_),seconds=time.perf_counter()-t0,vocabulary_size=len(fp['vectorizador'].vocabulary_),zero_train=int(np.sum(fp['vectorizador'].transform(normalized[cfg['normalization']][0]).getnnz(axis=1)==0)),zero_dev=int(np.sum(fp['vectorizador'].transform(normalized[cfg['normalization']][1]).getnnz(axis=1)==0)))
        records.append(row)
        interpretations.extend(experiment_interpretation(exp_id,cfg,fp,km,Xdv,dev,labels))
        pd.DataFrame(records).to_csv(OUT/'experiments.tsv',sep='\t',index=False)
        pd.DataFrame(interpretations).to_csv(OUT/'dev_cluster_interpretation.tsv',sep='\t',index=False)
```

Lectura de las líneas de este fragmento:

- Línea 141: Reúne en una fila los parámetros y resultados del experimento; los campos explican cómo se obtuvo su puntuación.

- Línea 142: Agrega un elemento a la colección que se exportará o presentará después.

- Línea 143: Continúa la expresión de la línea anterior; hay que leerla dentro de la llamada o estructura completa mostrada arriba.

- Línea 144: Exporta la tabla al archivo indicado. Esta es la línea que produce la evidencia TSV.

- Línea 145: Exporta la tabla al archivo indicado. Esta es la línea que produce la evidencia TSV.

Archivo y enlace exacto: [generar_informe.py](/Users/diamandazurita/Downloads/NLP_News_Project/fase_2_aprendizaje/no_supervisada/generar_informe.py:54). Función o contexto: `generate`. Líneas 54–65.

```python
    factors=['normalization','representation','ngram','k']
    controlled=[]
    factorial=exps[exps.stage.eq('factorial')]
    for factor in factors:
        others=[f for f in factors if f!=factor]
        for _, group in factorial.groupby(others):
            rows=list(group.to_dict('records'))
            for i,a in enumerate(rows):
                for b in rows[i+1:]:
                    controlled.append(dict(factor=factor,experiment_a=a['experiment_id'],experiment_b=b['experiment_id'],value_a=a[factor],value_b=b[factor],silhouette_a=a['silhouette'],silhouette_b=b['silhouette'],delta=b['silhouette']-a['silhouette'],eligible_a=a['eligible'],eligible_b=b['eligible']))
    pairs=pd.DataFrame(controlled)
    pairs.to_csv(OUT/'comparaciones_controladas.tsv',sep='\t',index=False)
```

Lectura de las líneas de este fragmento:

- Línea 54: Asigna el valor o resultado de la derecha a la variable de la izquierda para usarlo en los pasos siguientes.

- Línea 55: Asigna el valor o resultado de la derecha a la variable de la izquierda para usarlo en los pasos siguientes.

- Línea 56: Asigna el valor o resultado de la derecha a la variable de la izquierda para usarlo en los pasos siguientes.

- Línea 57: Repite el bloque para cada valor de la colección indicada, por ejemplo una configuración, semilla o grupo.

- Línea 58: Asigna el valor o resultado de la derecha a la variable de la izquierda para usarlo en los pasos siguientes.

- Línea 59: Repite el bloque para cada valor de la colección indicada, por ejemplo una configuración, semilla o grupo.

- Línea 60: Asigna el valor o resultado de la derecha a la variable de la izquierda para usarlo en los pasos siguientes.

- Línea 61: Repite el bloque para cada valor de la colección indicada, por ejemplo una configuración, semilla o grupo.

- Línea 62: Repite el bloque para cada valor de la colección indicada, por ejemplo una configuración, semilla o grupo.

- Línea 63: Agrega un elemento a la colección que se exportará o presentará después.

- Línea 64: Asigna el valor o resultado de la derecha a la variable de la izquierda para usarlo en los pasos siguientes.

- Línea 65: Exporta la tabla al archivo indicado. Esta es la línea que produce la evidencia TSV.

Archivo y enlace exacto: [generar_guia.py](/Users/diamandazurita/Downloads/NLP_News_Project/fase_2_aprendizaje/no_supervisada/generar_guia.py:37). Función o contexto: `consolidate_log`. Líneas 37–51.

```python
def consolidate_log():
    """Reúne los 91 candidatos y los tres ajustes de estabilidad sin reabrir selección."""
    import pandas as pd
    exp=pd.read_csv(OUT/'experiments.tsv',sep='\t')
    exp['candidate_for_selection']=True
    stability=json.loads((OUT/'estabilidad_dev.json').read_text())
    cfg=json.loads((OUT/'seleccion_dev.json').read_text())['config']
    cfg={**cfg,'ngram':str(tuple(cfg['ngram']))}
    additional=[]
    for row in stability['trials']:
        q={key:value for key,value in row.items() if key!='seed'}
        additional.append(dict(experiment_id=f"STAB{row['seed']:03d}",task='clustering',method='KMeans',stage='stability_dev',**cfg,**q,random_state=row['seed'],selected=False,candidate_for_selection=False,eligible=q['occupied_clusters']==cfg['k'] and q['min_share']>=.01 and q['max_share']<=.65))
    full=pd.concat([exp,pd.DataFrame(additional)],ignore_index=True)
    assert len(full)==94 and full.experiment_id.is_unique
    full.to_csv(OUT/'experiments_completo.tsv',sep='\t',index=False)
```

Lectura de las líneas de este fragmento:

- Línea 37: Define la función que se invoca desde otro punto del programa; definirla todavía no ejecuta su contenido.

- Línea 38: Continúa la expresión de la línea anterior; hay que leerla dentro de la llamada o estructura completa mostrada arriba.

- Línea 39: Continúa la expresión de la línea anterior; hay que leerla dentro de la llamada o estructura completa mostrada arriba.

- Línea 40: Lee una tabla del disco; sep=\t indica que sus columnas están separadas por tabuladores.

- Línea 41: Asigna el valor o resultado de la derecha a la variable de la izquierda para usarlo en los pasos siguientes.

- Línea 42: Asigna el valor o resultado de la derecha a la variable de la izquierda para usarlo en los pasos siguientes.

- Línea 43: Asigna el valor o resultado de la derecha a la variable de la izquierda para usarlo en los pasos siguientes.

- Línea 44: Asigna el valor o resultado de la derecha a la variable de la izquierda para usarlo en los pasos siguientes.

- Línea 45: Asigna el valor o resultado de la derecha a la variable de la izquierda para usarlo en los pasos siguientes.

- Línea 46: Repite el bloque para cada valor de la colección indicada, por ejemplo una configuración, semilla o grupo.

- Línea 47: Asigna el valor o resultado de la derecha a la variable de la izquierda para usarlo en los pasos siguientes.

- Línea 48: Registra o consulta la decisión de selección; no es la categoría original de la noticia.

- Línea 49: Asigna el valor o resultado de la derecha a la variable de la izquierda para usarlo en los pasos siguientes.

- Línea 50: Exige que la condición sea verdadera. Si no lo es, la ejecución se detiene con AssertionError.

- Línea 51: Exporta la tabla al archivo indicado. Esta es la línea que produce la evidencia TSV.

Para explicarlo con tus palabras: Guardamos todas las pruebas y utilizamos comparaciones controladas para explicar qué cambió, qué empeoró y qué no produjo diferencia.

## 20. Interfaz para consultar métricas, grupos y nuevas noticias

Qué pide el profesor: consultar sección(es) 10 del PDF. Resumen del requisito: Sin argumentos muestra el resumen y tamaños. --cluster N muestra términos y noticias. --texto aplica el modelo guardado a un título+descripción nuevo en inglés. Estos comandos no entrenan.

### Para entenderlo desde cero

La interfaz es un script de consola para consultar el trabajo sin reentrenarlo. Sin argumentos muestra configuración elegida, métricas finales y tamaños. --cluster seguido de un número muestra sus diez términos, las dos noticias y el caso de frontera.

--texto recibe un título+descripción en inglés y lo asigna con el modelo persistido. Utiliza la misma normalización y vocabulario final; no añade la noticia al corpus ni actualiza los centros. Que devuelva un cluster no garantiza que la noticia sea pertinente: vocabulario desconocido o texto demasiado corto pueden dificultar la interpretación.

Este archivo es el punto más sencillo para una demostración en clase. No necesitas ejecutar experimentos.py para enseñar una tabla; eso iniciaría un proceso de entrenamiento que además está protegido contra sobrescritura. La consulta normal solo carga TSV o el modelo, según la opción utilizada.

Ejemplo: Comando desde la raíz: .venv/bin/python fase_2_aprendizaje/no_supervisada/inspeccionar.py --cluster 2. Debe mostrar términos relacionados con Trump, elecciones y administración estadounidense, junto con las noticias exportadas.

### Dónde está y qué archivo demuestra el resultado

- inspeccionar.py: Lee TSV para mostrar métricas/grupos y carga el joblib al usar --texto. No actualiza el modelo.

- README.md: Punto de entrada para preparar el entorno, consultar resultados y reproducir en una copia limpia. No entrena por abrirlo.

### Cómo leer la implementación

- argparse define las opciones de la terminal.

- Si existe --texto, joblib.load carga el modelo y predict devuelve el grupo.

- Si existe --cluster, se filtran los TSV por cluster_id.

- Si no se pasa ninguna opción, se imprime el resumen.

- Un cluster inexistente produce un mensaje de error de uso.

Archivo y enlace exacto: [inspeccionar.py](/Users/diamandazurita/Downloads/NLP_News_Project/fase_2_aprendizaje/no_supervisada/inspeccionar.py:14). Función o contexto: `nivel principal del módulo`. Líneas 14–20.

```python
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--cluster',type=int,help='Ver términos y noticias del grupo indicado.')
parser.add_argument('--texto',help='Asignar una noticia nueva (título + descripción) al modelo final.')
args = parser.parse_args()
if args.texto:
    model = joblib.load(HERE/'modelos/kmeans_pipeline.joblib')
    print('Cluster:',int(model.predict([args.texto])[0]))
```

Lectura de las líneas de este fragmento:

- Línea 14: Asigna el valor o resultado de la derecha a la variable de la izquierda para usarlo en los pasos siguientes.

- Línea 15: Asigna el valor o resultado de la derecha a la variable de la izquierda para usarlo en los pasos siguientes.

- Línea 16: Asigna el valor o resultado de la derecha a la variable de la izquierda para usarlo en los pasos siguientes.

- Línea 17: Asigna el valor o resultado de la derecha a la variable de la izquierda para usarlo en los pasos siguientes.

- Línea 18: Evalúa la condición y ejecuta esta rama únicamente cuando corresponde.

- Línea 19: Carga el modelo persistido; por sí mismo no ajusta ni cambia los centros.

- Línea 20: Muestra información al ejecutar la consola; no cambia el ajuste del modelo.

Archivo y enlace exacto: [inspeccionar.py](/Users/diamandazurita/Downloads/NLP_News_Project/fase_2_aprendizaje/no_supervisada/inspeccionar.py:21). Función o contexto: `nivel principal del módulo`. Líneas 21–34.

```python
elif args.cluster is not None:
    words = pd.read_csv(HERE/'resultados/top_words.tsv',sep='\t')
    reps = pd.read_csv(HERE/'resultados/representative_documents.tsv',sep='\t')
    selected = words[words.cluster_id.eq(args.cluster)]
    if selected.empty:
        parser.error('El cluster no existe. Consulta primero el resumen sin argumentos.')
    print('Términos:',', '.join(selected.term))
    for _, row in reps[reps.cluster_id.eq(args.cluster)].iterrows():
        print(f'\n{row.role}: {row.document_id} ({row.split})\n{row.text}\n{row.url}')
else:
    print((HERE/'resultados/seleccion_dev.json').read_text())
    print(pd.read_csv(HERE/'resultados/metricas_finales.tsv',sep='\t').to_string(index=False))
    print('\nTamaños por partición:')
    print(pd.read_csv(HERE/'resultados/cluster_sizes.tsv',sep='\t').to_string(index=False))
```

Lectura de las líneas de este fragmento:

- Línea 21: Evalúa la condición y ejecuta esta rama únicamente cuando corresponde.

- Línea 22: Lee una tabla del disco; sep=\t indica que sus columnas están separadas por tabuladores.

- Línea 23: Lee una tabla del disco; sep=\t indica que sus columnas están separadas por tabuladores.

- Línea 24: Registra o consulta la decisión de selección; no es la categoría original de la noticia.

- Línea 25: Registra o consulta la decisión de selección; no es la categoría original de la noticia.

- Línea 26: Continúa la expresión de la línea anterior; hay que leerla dentro de la llamada o estructura completa mostrada arriba.

- Línea 27: Registra o consulta la decisión de selección; no es la categoría original de la noticia.

- Línea 28: Repite el bloque para cada valor de la colección indicada, por ejemplo una configuración, semilla o grupo.

- Línea 29: Filtra, conserva o comprueba la partición de los documentos; no la convierte en característica del texto.

- Línea 30: Ejecuta la alternativa cuando la condición anterior no se cumple.

- Línea 31: Muestra información al ejecutar la consola; no cambia el ajuste del modelo.

- Línea 32: Lee una tabla del disco; sep=\t indica que sus columnas están separadas por tabuladores.

- Línea 33: Muestra información al ejecutar la consola; no cambia el ajuste del modelo.

- Línea 34: Lee una tabla del disco; sep=\t indica que sus columnas están separadas por tabuladores.

Para explicarlo con tus palabras: La consulta permite mostrar resultados existentes y asignar un texto nuevo sin volver a entrenar, con las mismas transformaciones del modelo final.

## 21. Versiones, semillas, datos y artefactos reproducibles

Qué pide el profesor: consultar sección(es) 3, 9 y 11 del PDF. Resumen del requisito: Se guardan versiones exactas, modelo spaCy, huella del corpus, parámetros finales, vocabulario y matrices léxicas por partición. El orden de filas de cada NPZ corresponde al TSV filtrado por split. El ZIP incorpora los binarios ignorados por Git.

### Para entenderlo desde cero

Reproducibilidad significa poder saber qué datos, versiones y decisiones produjeron los resultados. requirements-lock.txt guarda versiones de las dependencias. configuracion_final.json registra el conjunto usado para reajustar, la configuración ganadora, la semilla y las versiones. parametros_pipeline_final.json amplía los parámetros completos, incluidos valores por defecto.

Los archivos NPZ son matrices dispersas: guardan eficientemente muchos ceros sin convertirse en una tabla enorme de texto. Para interpretarlas necesitas el orden de documentos y el vocabulario. La fila i de cada NPZ corresponde a la fila i del TSV después de filtrar por esa partición; la columna j corresponde al término j del vocabulario.

El modelo joblib guarda los objetos ajustados, no solo los nombres de los parámetros. manifest.json contiene tamaños y huellas SHA-256 para comprobar integridad. Es distinto de un resultado científico: una huella permite detectar un cambio de bytes, pero no demuestra por sí sola que el método sea adecuado.

Ejemplo: Si se cambia el orden de las noticias al leer una matriz sin actualizar sus IDs, podrías atribuir el vector de N2256 a otra noticia. Por eso la auditoría compara todas las matrices y su orden, no solo su número de filas.

### Dónde está y qué archivo demuestra el resultado

- requirements-lock.txt: Versiones de bibliotecas y URL con huella del modelo inglés de spaCy. Es la receta del entorno utilizado, no un resultado del modelo.

- resultados/configuracion_final.json: Guarda configuración ganadora, 3948 textos de ajuste, versiones, semilla y test_used_for_selection=False.

- resultados/parametros_pipeline_final.json: Incluye los parámetros de cada etapa del modelo cargado, también opciones por defecto del vectorizador.

- resultados/manifest.json: Ruta, tamaño y SHA-256 de los archivos entregados, excepto el propio manifiesto y el ZIP.

- modelos/X_words_train.npz: Representación de las 3251 noticias train con el vocabulario del reajuste final. Todavía está en columnas de términos, antes de SVD y L2.

- modelos/X_words_dev.npz: Representación de las 697 noticias dev con el vocabulario final. Dev ya participó en el reajuste de ese modelo.

- modelos/X_words_test.npz: Representación de las 697 noticias reservadas, usando el vocabulario aprendido en train+dev.

- ejecucion.log: Salida de la ejecución de aprendizaje, con IDs de experimentos, configuraciones, tamaños y finalización. Puede contener avisos no fatales del entorno.

### Cómo leer la implementación

- joblib.dump conserva los componentes entrenados del pipeline.

- save_npz almacena las matrices léxicas por split.

- get_feature_names_out exporta el vocabulario en orden.

- SHA-256 vincula los datos usados con la configuración registrada.

- El ZIP incluye los binarios aunque Git los ignore.

Archivo y enlace exacto: [experimentos.py](/Users/diamandazurita/Downloads/NLP_News_Project/fase_2_aprendizaje/no_supervisada/experimentos.py:193). Función o contexto: `run`. Líneas 193–193.

```python
    write_json(OUT/'configuracion_final.json',dict(selected_experiment=winner['experiment_id'],config=cfg,train_dev_count=len(train_dev),test_used_for_selection=False,vectorizer_refitted_on='train+dev',source_sha256=plan['source_sha256'],python=platform.python_version(),versions={p:version(p) for p in ['numpy','pandas','scipy','scikit-learn','spacy','en-core-web-sm','joblib','reportlab','matplotlib','threadpoolctl']},seed=SEED))
```

Lectura de las líneas de este fragmento:

- Línea 193: Guarda en JSON la configuración, decisión o información estructurada que se le entrega.

Archivo y enlace exacto: [experimentos.py](/Users/diamandazurita/Downloads/NLP_News_Project/fase_2_aprendizaje/no_supervisada/experimentos.py:207). Función o contexto: `run`. Líneas 207–209.

```python
    for split in ['train','dev','test']:
        mask = docs.split.eq(split).to_numpy()
        q = quality(X[mask],labels[mask],cfg['k'])
```

Lectura de las líneas de este fragmento:

- Línea 207: Filtra, conserva o comprueba la partición de los documentos; no la convierte en característica del texto.

- Línea 208: Filtra, conserva o comprueba la partición de los documentos; no la convierte en característica del texto.

- Línea 209: Asigna el valor o resultado de la derecha a la variable de la izquierda para usarlo en los pasos siguientes.

Archivo y enlace exacto: [experimentos.py](/Users/diamandazurita/Downloads/NLP_News_Project/fase_2_aprendizaje/no_supervisada/experimentos.py:235). Función o contexto: `run`. Líneas 235–235.

```python
        sparse.save_npz(MOD/f'X_words_{split}.npz',Xwords[mask])
```

Lectura de las líneas de este fragmento:

- Línea 235: Guarda la matriz dispersa de esa partición, conservando el orden de filas y columnas.

Para explicarlo con tus palabras: Entregamos datos, código, entorno, configuración, vocabulario, matrices y modelo; así se puede rastrear y reproducir el proceso, no solamente ver un número final.

## 22. Informe PDF con figuras, nombres y limitaciones

Qué pide el profesor: consultar sección(es) 11 y 12 del PDF. Resumen del requisito: El informe analítico documenta resultados y limitaciones. La guía explica cada paso para estudiarlo. Todo el trabajo se reconoce como elaboración conjunta de los integrantes del equipo, conforme a la aclaración de la usuaria.

### Para entenderlo desde cero

El informe analítico y la guía de estudio tienen propósitos distintos. El informe explica el método, los resultados obtenidos y sus limitaciones. Esta guía enseña cómo funciona cada paso y cómo encontrarlo en el código. Ambos se generan a partir de los archivos reales y reconocen el trabajo de todo el equipo.

El informe incluye gráficas de tamaños y experimentos, comparación de configuraciones, interpretación de los doce grupos y ejemplos de eventos. La lectura temática está escrita en interpretacion_manual.json, mientras que métricas, términos y noticias provienen de TSV. El generador junta esas fuentes y crea Markdown y PDF.

Generar un PDF no equivale a volver a ejecutar el aprendizaje. generar_informe.py lee resultados existentes y dibuja; no cambia los clusters ni reajusta K-Means. Si se modifican los modelos en otra corrida, la interpretación temática debe revisarse antes de reutilizarla, porque el mismo número de cluster puede significar otra cosa.

Ejemplo: La gráfica de tamaños se construye con cluster_sizes.tsv. El párrafo que llama al cluster 3 un sesgo de publicaciones científicas sale de interpretacion_manual.json y se acompaña con las noticias reales guardadas.

### Dónde está y qué archivo demuestra el resultado

- resultados/INFORME_NO_SUPERVISADO.pdf: Versión de presentación con gráficas y evidencia real.

- resultados/figuras/experimentos_dev.png: Silhouette dev por experimento, diferenciando admisibles y elegido.

- resultados/figuras/tamanos_clusters.png: Barras apiladas de train/dev/test para los doce grupos.

### Cómo leer la implementación

- El generador lee tablas, configuración e interpretación.

- matplotlib produce las figuras PNG.

- Cada apartado se añade a una lista de texto.

- El archivo Markdown conserva la versión editable.

- ReportLab construye el PDF, con autoría conjunta del equipo.

Archivo y enlace exacto: [generar_informe.py](/Users/diamandazurita/Downloads/NLP_News_Project/fase_2_aprendizaje/no_supervisada/generar_informe.py:162). Función o contexto: `generate`. Líneas 162–163.

```python
    md='\n\n'.join(lines)+'\n'
    (OUT/'INFORME_NO_SUPERVISADO.md').write_text(md,encoding='utf-8')
```

Lectura de las líneas de este fragmento:

- Línea 162: Asigna el valor o resultado de la derecha a la variable de la izquierda para usarlo en los pasos siguientes.

- Línea 163: Asigna el valor o resultado de la derecha a la variable de la izquierda para usarlo en los pasos siguientes.

Archivo y enlace exacto: [generar_informe.py](/Users/diamandazurita/Downloads/NLP_News_Project/fase_2_aprendizaje/no_supervisada/generar_informe.py:181). Función o contexto: `generate`. Líneas 181–181.

```python
    SimpleDocTemplate(str(OUT/'INFORME_NO_SUPERVISADO.pdf'),pagesize=(21*cm,29.7*cm),rightMargin=1.5*cm,leftMargin=1.5*cm,topMargin=1.5*cm,bottomMargin=1.7*cm).build(story,onFirstPage=footer,onLaterPages=footer)
```

Lectura de las líneas de este fragmento:

- Línea 181: Construye el PDF a partir de texto, figuras y estilos, sin volver a entrenar el modelo.

Para explicarlo con tus palabras: El informe presenta resultados y limitaciones; la guía sirve para estudiarlos. Los dos reconocen la elaboración conjunta de todos los integrantes.

## 23. Auditoría ejecutada y protección de la corrida final

Qué pide el profesor: consultar sección(es) 4, 9, 11 y 13 del PDF. Resumen del requisito: validar_resultados.py comprueba integridad general. auditar_cumplimiento.py agregó 39 comprobaciones independientes de datos, particiones, parámetros reales, matrices completas, asignaciones, métricas, vocabulario y representantes. No se reentrenó K-Means ni se cambió el ganador. La protección impide sobrescribir automáticamente la corrida final.

### Para entenderlo desde cero

Validar resultados es comprobar que los archivos de la entrega concuerden entre sí. validar_resultados.py verifica IDs, tamaños, cantidad de términos y representantes, selección y carga del modelo. auditar_cumplimiento.py hace una revisión más profunda: reconstruye los splits, revisa las matrices completas y recalcula métricas de las mismas asignaciones.

La auditoría también reconstruye un vectorizador sobre train+dev para verificar el vocabulario final y calcula qué noticias son realmente las más cercanas a cada centro. Esto ayuda a detectar errores silenciosos, por ejemplo exportar representantes de otro grupo o columnas desalineadas. No entrena de nuevo K-Means ni decide una nueva configuración con test.

El programa principal se detiene si ya existe configuracion_final.json. Esa protección evita que, por intentar consultar resultados, sobrescribas el modelo final y repitas inadvertidamente el estudio. Las comprobaciones aprobadas respaldan consistencia técnica, no prometen que todos los temas sean puros: el análisis de calidad sigue siendo necesario.

Ejemplo: Si alguien cambiara manualmente cluster_id de una noticia en clustering_results.tsv, el modelo recargado podría producir otro grupo y la auditoría fallaría. Si borrara una noticia, dejarían de coincidir IDs y sumas.

### Dónde está y qué archivo demuestra el resultado

- validar_resultados.py: Comprueba correspondencia de IDs, sumas, selección, representantes y persistencia del modelo. Es más breve que la auditoría profunda.

- auditar_cumplimiento.py: Reconstruye el reparto; verifica las 2157 noticias originales; compara matrices completas, clusters, métricas, vocabulario y representantes. Guarda parámetros y 39 controles aprobados.

- resultados/validacion.txt: Mensaje OK de validar_resultados.py con conteos y consistencia comprobada.

- resultados/auditoria_cumplimiento.json: Lista los controles técnicos con estado y evidencia; incluye alcance y autoría conjunta.

### Cómo leer la implementación

- assert y check detienen la auditoría cuando una condición no se cumple.

- np.array_equal exige que coincidan todas las asignaciones comprobadas.

- np.isclose compara métricas permitiendo únicamente tolerancia numérica pequeña.

- Se escriben los controles aprobados en auditoria_cumplimiento.json.

- La guardia de configuracion_final.json impide una sobrescritura automática.

Archivo y enlace exacto: [experimentos.py](/Users/diamandazurita/Downloads/NLP_News_Project/fase_2_aprendizaje/no_supervisada/experimentos.py:102). Función o contexto: `run`. Líneas 102–103.

```python
    if (OUT / 'configuracion_final.json').exists():
        raise SystemExit('Ya existe una corrida final. Usa una copia de la carpeta para un nuevo estudio; no se reabre test automáticamente.')
```

Lectura de las líneas de este fragmento:

- Línea 102: Evalúa la condición y ejecuta esta rama únicamente cuando corresponde.

- Línea 103: Detiene el proceso y comunica la condición que impide continuar.

Archivo y enlace exacto: [auditar_cumplimiento.py](/Users/diamandazurita/Downloads/NLP_News_Project/fase_2_aprendizaje/no_supervisada/auditar_cumplimiento.py:67). Función o contexto: `run`. Líneas 67–69.

```python
        check('Modelo y asignaciones '+split,np.array_equal(model['kmeans'].predict(latent),assigned),'El artefacto persistido reproduce las asignaciones de la matriz guardada')
        score=silhouette_score(latent,assigned,metric='euclidean')
        check('Silhouette guardado '+split,np.isclose(score,metrics.loc[split,'silhouette'],atol=1e-12),'Recálculo de la misma métrica sobre las mismas asignaciones; sin modificar parámetros')
```

Lectura de las líneas de este fragmento:

- Línea 67: Filtra, conserva o comprueba la partición de los documentos; no la convierte en característica del texto.

- Línea 68: Calcula separación geométrica sobre los vectores y grupos indicados; la distancia se fija en euclidean.

- Línea 69: Filtra, conserva o comprueba la partición de los documentos; no la convierte en característica del texto.

Para explicarlo con tus palabras: Las 39 comprobaciones aprobadas verifican integridad y reproducibilidad de lo entregado; la evaluación temática y sus límites se presentan por separado.

## Cómo leer los archivos con resultados: ejemplos reales

Ejemplo de asignación real: document_id=N2256; split=train; cluster_id=2. document_id identifica la noticia, split indica su partición y cluster_id el grupo aprendido. method/configuration describe la configuración; no es la etiqueta gold.

Ejemplo de tamaño real: cluster 0 tiene 680 train, 131 dev y 140 test: 951 en total. Es el mismo modelo final que generó las asignaciones.

Ejemplo de términos del cluster 1: ai, nvidia, company, huang, jensen, jensen huang, ceo, data, intelligence, spy. Son características de texto: algunos términos tienen más de una palabra.

Ejemplo de una fila de experiments.tsv: KM083 tiene normalization=sin_stopwords, representation=binaria, ngram=(1,2), svd=50, k=12, n_init=10, init=k-means++, max_iter=300, random_state=0, eligible=True y selected=True. No significa que esos sean los únicos valores probados.

Diferencia de métricas que debes recordar: seleccion_dev.json contiene dev de selección 0.165800; metricas_finales.tsv contiene train 0.155980, dev 0.154702 y test 0.177708 del modelo ya reajustado. El dev reajustado no se usa para justificar el ganador.

## Comandos de estudio y demostración

Ejecutar desde la raíz NLP_News_Project, con el entorno ya preparado:

```bash
.venv/bin/python fase_2_aprendizaje/no_supervisada/inspeccionar.py
.venv/bin/python fase_2_aprendizaje/no_supervisada/inspeccionar.py --cluster 2
.venv/bin/python fase_2_aprendizaje/no_supervisada/validar_resultados.py
.venv/bin/python fase_2_aprendizaje/no_supervisada/auditar_cumplimiento.py
```

Primer comando: resumen. Segundo: términos y noticias de un grupo. Tercero: validación breve. Cuarto: auditoría profunda. No necesitas ejecutar experimentos.py para estudiar; eso corresponde a una corrida nueva en una copia limpia.

## Preguntas para practicar antes de exponer

Pregunta: ¿Dónde se valida que el texto esté bien construido?

Respuesta: En experimentos.py, load_documents(): cruza IDs y compara text con title + espacio + description. El apartado 1 muestra el código exacto y sus líneas.

Pregunta: ¿Dónde se comprueba el reparto?

Respuesta: En auditar_cumplimiento.py, run(): reconstruye los dos train_test_split y exige que los IDs de cada split coincidan. El entrenamiento reutiliza la columna split.

Pregunta: ¿Dónde se evalúa con dev?

Respuesta: En la función anidada trial de experimentos.py: ajusta con Xtr, predice Xdv y llama quality(Xdv, labels, k). No es lo mismo que validar integridad del TSV.

Pregunta: ¿Dónde se reentrena?

Respuesta: En experimentos.py: final.fit(train_dev.text). Es el reajuste posterior a la selección, no un nuevo barrido de parámetros.

Pregunta: ¿Por qué conserva category si es no supervisado?

Respuesta: Para estratificación original y análisis externo. fit(Xtr) no recibe category ni y, y best() no usa ARI/NMI contra las categorías.

Pregunta: ¿Por qué doce grupos si hay diez categorías?

Respuesta: Porque las categorías no fijan el objetivo de K-Means. Se compararon varios k y doce ganó bajo el criterio dev declarado.

Pregunta: ¿Qué significan 50, 12 y 10 en el ganador?

Respuesta: 50 son dimensiones SVD; 12 son clusters; 10 son reinicios n_init. No son la misma cantidad.

Pregunta: ¿Silhouette 0.1777 es 17.77% de precisión?

Respuesta: No. Es una medida de separación geométrica; aquí indica separación positiva pero modesta, con mezcla temática.

Pregunta: ¿Dónde están las dos noticias por grupo?

Respuesta: En resultados/representative_documents.tsv, role=representativo. Hay dos por grupo, más un caso role=frontera adicional.

Pregunta: ¿Por qué un cluster puede ser de formato?

Respuesta: Expresiones repetidas como Continue reading o DOI acercan textos de temas diferentes. La variante sin_stopwords no elimina automáticamente esas fórmulas.

Pregunta: ¿Los términos se sacan directamente de SVD?

Respuesta: Se reconstruyen los centroides con inverse_transform y se relacionan con las columnas del vocabulario original; no se muestran índices latentes como palabras.

Pregunta: ¿Por qué hay dos valores de silhouette dev?

Respuesta: 0.165800 corresponde a selección con modelo ajustado solo con train. 0.154702 corresponde al modelo final reajustado con train+dev; ese dev ya no es independiente.

Pregunta: ¿ARI de estabilidad y ARI externo son lo mismo?

Respuesta: La fórmula es la misma, pero las particiones comparadas cambian: dos ejecuciones en estabilidad; clusters frente a categorías en análisis externo.

Pregunta: ¿Para mostrar resultados ejecuto experimentos.py?

Respuesta: No. Se usa inspeccionar.py o se abren los TSV y el PDF. experimentos.py está destinado a entrenar una corrida nueva.

Pregunta: ¿La captura de 50 pruebas describe el modelo nuevo?

Respuesta: No. Describe el módulo anterior ../agrupamiento. La entrega actual está en no_supervisada y tiene 91 candidatos, doce grupos y otros parámetros.

Pregunta: ¿Quién hizo el trabajo?

Respuesta: Todos los integrantes del equipo colaboraron en su elaboración. La autoría conjunta aparece en el informe y la guía.

## Un guion breve para contar el trabajo del equipo

Nuestro equipo reutilizó las noticias y las particiones de la etapa anterior. Comparamos tres normalizaciones, tres representaciones y tres rangos de n-gramas con K-Means, sin usar categorías para entrenar. Seleccionamos con dev mediante silhouette y controles de tamaño; ganó una representación binaria sin stopwords, con unigramas y bigramas, SVD de 50 dimensiones y doce grupos. Después reajustamos el pipeline completo con train+dev y evaluamos test sin volver a elegir parámetros. Exportamos tamaños, diez términos y dos noticias por grupo, estudiamos estabilidad y eventos y explicamos tanto los temas reconocibles como los grupos contaminados por formato editorial.

## Qué corresponde a esta rama y qué va en el informe general

Esta guía estudia K-Means, la tarea B elegida por el equipo. LDA y perplexity pertenecen a otra rama opcional. Macro-F1, accuracy y los modelos Naive Bayes, regresión logística y SVC corresponden a supervisada. La entrega completa de etapa II integra ambas tareas en el informe general. La autoría de esta parte es conjunta de todos los integrantes. Los grupos mixtos y el silhouette modesto son resultados analizados, no garantías de que existan doce temas puros.
