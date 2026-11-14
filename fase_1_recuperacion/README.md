# Fase 1 · Recuperación de Información y Modelos de Espacio Vectorial (VSM)

### Integrantes:
- Hernandez Espinoza Ricardo
- Espinoza Diaz Diana Karen
- Robles Lozano Byron
- Saenz Buttanda Hector Arturo
- Zurita Salinas Diamanda Abigail

El módulo de Fase 1 se estructura en tres componentes clave:

```text
fase_1_recuperacion/
├── scraper/              # Motor RSS, recolección de noticias y base de datos
├── normalizacion/        # Limpieza lingüística, lematización con spaCy y POS-filtering
└── benchmark/            # Modelos VSM (31 configuraciones), evaluación IR y CLI
```

## Instalación

```powershell
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Flujo de Trabajo

Desde la raíz del proyecto (`Proyecto PLN/`):

```powershell
# 1. Ejecutar normalización del corpus
python "fase_1_recuperacion/normalizacion/normalize_corpus.py"

# 2. Construir los modelos de espacio vectorial (VSM)
python "fase_1_recuperacion/benchmark/codigo/vector_space_models.py"

# 3. Realizar búsquedas de prueba
python "fase_1_recuperacion/benchmark/main.py" search --query "artificial intelligence regulation"

# 4. Ejecutar el benchmark completo (evaluación con métricas P@10, R@10, F1@10, RR, nDCG@10)
python "fase_1_recuperacion/benchmark/main.py" benchmark
```

## Estructura de Salidas y Juicios

* Las consultas (16) y juicios de relevancia se ubican en `fase_1_recuperacion/benchmark/data/`.
* Las salidas activas se almacenan en `fase_1_recuperacion/benchmark/resultados/`:
  * `rankings.tsv`: ranking por consulta.
  * `evaluation_results.tsv`: métricas detalladas por consulta.
  * `comparison.tsv`: tabla comparativa de configuraciones.
