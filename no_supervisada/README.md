# K-Means: aprendizaje no supervisado de noticias

Implementación y evidencia de la Tarea B de `Proyecto_01_Noticias_Etapa_II.pdf`. Se conserva el módulo anterior `../agrupamiento/` como referencia; esta carpeta contiene una nueva ejecución independiente.

Consulta `ESTADO_ENTREGA.md` para ver requisito por requisito y qué queda para integrar la etapa completa. El paquete `entrega_no_supervisada.zip` incluye las salidas y los binarios locales.

## Guía para estudiar y exponer

Abre primero [GUIA_DE_ESTUDIO_NO_SUPERVISADO.pdf](GUIA_DE_ESTUDIO_NO_SUPERVISADO.pdf). Explica el recorrido desde cero, un glosario, qué hace cada archivo, el vínculo con los archivos de la captura anterior, 23 requisitos con 44 fragmentos reales y explicaciones de líneas, ejemplos y preguntas de repaso.

[La versión Markdown](GUIA_DE_ESTUDIO_NO_SUPERVISADO.md) permite copiar código exacto y abrir enlaces a las líneas. Las versiones `GUIA_REQUISITOS_Y_CODIGO.*` conservan el mismo contenido actualizado para mantener los enlaces anteriores. El material de estudio se guarda en `contenido_estudio.json` y `material_apoyo_estudio.py`; no interviene en el entrenamiento.

La autoría del trabajo se reconoce conjuntamente a todos los integrantes del equipo. El informe analítico de resultados sigue en `resultados/INFORME_NO_SUPERVISADO.pdf`.

`resultados/auditoria_cumplimiento.json` documenta 39 comprobaciones técnicas; `resultados/experiments_completo.tsv` consolida los 91 candidatos y los tres ajustes de estabilidad.

## Archivos de entrada y código

- `datos/`: copia exacta del corpus original de 4,645 noticias y del conjunto etiquetado con las particiones existentes (3,251 train, 697 dev, 697 test).
- `soporte/normalizacion.py`: copia sin cambios del normalizador del pipeline del equipo, para conservar las mismas operaciones y permitir compartir esta carpeta.
- `experimentos.py`: comparaciones controladas, selección en dev, estabilidad, reajuste completo con train + dev y evidencia final de test.
- `inspeccionar.py`: interfaz de consulta de métricas/grupos y asignación de noticias nuevas.
- `validar_resultados.py`: verifica integridad de las salidas y persistencia del modelo.
- `auditar_cumplimiento.py`: comprueba particiones, matrices completas, vocabulario, métricas y selección sin reentrenar K-Means.
- `generar_guia.py`: extrae código real y verifica las líneas de esta guía.
- `generar_informe.py`: regenera el informe PDF y las figuras a partir de resultados existentes.
- `interpretacion_manual.json`: lectura cualitativa de palabras, noticias y casos de frontera; no se utiliza para entrenar el modelo.
- `requirements-lock.txt`: versiones exactas del entorno utilizado, incluido el modelo inglés de spaCy.
- `referencias/Proyecto_01_Noticias_Etapa_II.pdf`: documento de requisitos proporcionado.

## Entregables de la ejecución

- `resultados/INFORME_NO_SUPERVISADO.pdf`: informe de esta tarea con metodología, comparaciones, figuras, temas y limitaciones.
- `resultados/experiments.tsv`: bitácora de todas las configuraciones probadas, métricas dev y decisión de selección.
- `resultados/clustering_results.tsv`: identificador, partición, grupo y configuración para cada noticia.
- `resultados/cluster_sizes.tsv`: tamaños por train/dev/test y total.
- `resultados/top_words.tsv`: diez términos por grupo en el espacio léxico original.
- `resultados/representative_documents.tsv`: dos noticias representativas y un caso de frontera por grupo, con texto y URL.
- `resultados/dev_cluster_interpretation.tsv`: términos y noticias dev por grupo para cada experimento.
- `resultados/metricas_finales.tsv`, `contingencia_*.tsv`: silhouette final y análisis externo con categorías (ARI/NMI).
- `resultados/eventos_recurrentes.tsv`: ejemplos trazables de coberturas de un mismo evento que se agrupan o se separan.
- `resultados/estabilidad_dev.json`, `comparaciones_controladas.tsv`, `resumen_factores.tsv`: sensibilidad a inicialización y cambios de factores.
- `resultados/plan_experimental.json`, `seleccion_dev.json`, `configuracion_final.json`: parámetros, criterio, versiones y huella del corpus.
- `resultados/labeled_documents.tsv`, `corpus_normalizado_final.tsv`, `vocabulario.tsv`: trazabilidad documental.
- `modelos/kmeans_pipeline.joblib`: pipeline final entrenado, incluido normalizador, vectorizador, SVD si se seleccionó, L2 y K-Means.
- `modelos/X_words_*.npz`: matrices léxicas del modelo final; el orden de filas es el del TSV de documentos filtrado por la partición correspondiente.
- `ejecucion.log`, `resultados/validacion.txt`: evidencia de ejecución y comprobaciones.

Las categorías nunca son características ni objetivos de ajuste. Solo se usan para análisis externo después de congelar el modelo. No se fuerza que k sea igual al número de categorías. Se usa únicamente K-Means: no hace falta implementar LDA.

## Consultar lo ya ejecutado

Desde la raíz del proyecto:

```bash
.venv/bin/python fase_2_aprendizaje/no_supervisada/inspeccionar.py
.venv/bin/python fase_2_aprendizaje/no_supervisada/inspeccionar.py --cluster 0
.venv/bin/python fase_2_aprendizaje/no_supervisada/inspeccionar.py --texto 'Título y descripción de una noticia en inglés'
.venv/bin/python fase_2_aprendizaje/no_supervisada/validar_resultados.py
```

## Reproducir en otro entorno

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r fase_2_aprendizaje/no_supervisada/requirements-lock.txt
```

El entorno utilizado es Python 3.13.5 en macOS. Para volver a entrenar, usa una copia de la carpeta de código con `datos/` y `soporte/`, sin `resultados/` ni `modelos/` de una corrida anterior:

```bash
.venv/bin/python -u fase_2_aprendizaje/no_supervisada/experimentos.py
.venv/bin/python fase_2_aprendizaje/no_supervisada/validar_resultados.py
.venv/bin/python fase_2_aprendizaje/no_supervisada/generar_informe.py
```

La protección contra sobrescritura evita repetir automáticamente la evaluación final. Regenerar el PDF o consultar los TSV no vuelve a entrenar ni a seleccionar usando test. La interpretación manual debe revisarse si una nueva ejecución cambia los grupos.

Los binarios `*.joblib` y `*.npz` están ignorados por el `.gitignore` general: existen localmente, pero **al compartir la entrega hay que incluir la carpeta completa**, no solamente los archivos que Git registra. El normalizador conservado en `soporte/` evita depender del módulo externo al cargar el modelo.

## Qué falta para entregar toda la etapa II

Esta carpeta cubre K-Means y su evidencia. El equipo aún debe integrar este informe con la clasificación supervisada y revisar la reproducibilidad del módulo supervisado (matrices ausentes en la copia inicial, corrida fija y reajuste del vectorizador final). El informe general anterior debe actualizarse con estos resultados; las métricas de la ejecución vieja de agrupamiento no describen este modelo.


Autoría: el trabajo fue realizado conjuntamente por todos los integrantes del equipo, según confirmación de la usuaria.
